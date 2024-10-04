
import logging
import multiprocessing as mp
from functools import partial
import numpy as np
import healpy as hp
from pathlib import Path
from typing import Tuple

from astropy import constants as const, units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value


class SheetMapper:
    """Handles sheet mapping operations for cosmological data visualization."""

    def __init__(self, nside: int = 8192):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.maps = {}

    def new_map(self, map_name: str, dtype: str = 'float32') -> None:
        """Create a new map with the given name and data type."""
        self.maps[map_name] = np.zeros(self.npix, dtype=dtype)

    def add_sheet_to_map(
        self,
        map_name: str,
        sheet: np.ndarray,
        wlen: callable,
        chi1: float,
        chi2: float,
        cosmology,
        zs: float
    ) -> None:
        """Add a sheet to the map using a weak lensing integral."""
        chi = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)  # [Mpc]
        dchi = chi1 - chi2  # [Mpc]
        wlen_integral = wlen(chi, cosmology, zs) * dchi  # [1]
        self.maps[map_name] += wlen_integral * sheet

class WeakLensingHelper:
    @staticmethod
    def wlen_chi_kappa(chi: float, cosmo: FlatLambdaCDM, zs: float) -> float:
        """Compute the weight function for weak lensing convergence."""
        chis = cosmo.comoving_distance(zs).value # Mpc
        H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
        z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
        dchi = np.clip(1 - chi / chis, 0, None)
        return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi # 1/Mpc
    
    @staticmethod
    def index_to_chi(index: int, cosmo: FlatLambdaCDM) -> Tuple[float, float]:
        """Convert an index to a comoving distance."""
        a1, a2 = 0.01 * index, 0.01 * (index + 1)
        z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * cosmo.h
        return chi1, chi2

class KappaConstructor:
    def __init__(self, datadir, zs_list=None, overwrite=False):
        self.datadir = Path(datadir)
        self.zs_list = zs_list or [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self._initialize_directories()
        
    def _initialize_directories(self):
        self.outputdir = self.datadir / "kappa"
        self.outputdir.mkdir(exist_ok=True)

        self.massdir = self.datadir / "mass_sheets"
        self.sheet_files = sorted(self.massdir.glob("delta-sheet-*.fits"))

    def compute_kappa(self):
        for zs in self.zs_list:
            logging.info(f"Starting computation for zs={zs}.")
            kappa_file = self.outputdir / f"kappa_zs{zs}.fits"
            if kappa_file.exists() and not self.overwrite:
                logging.info(f"Kappa for zs={zs} exists. Skipping.")
                continue
            kappa = self._compute_kappa(zs)
            hp.write_map(str(kappa_file), kappa, dtype=np.float32)
            logging.info(f"Kappa saved to {kappa_file.name}.")

    def _compute_kappa(self, zs: float) -> np.ndarray:
        """Compute the global kappa map."""
        mapper = SheetMapper()
        mapper.new_map("kappa")
        process_partial = partial(self._process_delta_sheet, zs=zs, mapper=mapper)
        
        # Use multiprocessing Pool to parallelize the work
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(process_partial, self.sheet_files)

        return np.sum(results, axis=0)

    def _process_delta_sheet(self, data_path: str, zs: float, mapper: SheetMapper) -> np.ndarray:
        """Process a single delta sheet and update the mapper."""
        i = int(data_path.split("-")[-1].split(".")[0])
        logging.info(f"Processing delta sheet index {i}")
        delta = hp.read_map(data_path).astype('float32')
        chi1, chi2 = WeakLensingHelper.index_to_chi(i, self.cosmo)
        mapper.add_sheet_to_map(
            "kappa",
            delta,
            WeakLensingHelper.wlen_chi_kappa,
            chi1,
            chi2,
            self.cosmo,
            zs
        )
        return mapper.maps["kappa"]