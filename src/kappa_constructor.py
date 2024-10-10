
import logging
from pathlib import Path
from typing import List, Tuple

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import constants as const, units as u
from multiprocessing import Pool

from src.info_extractor import InfoExtractor

def process_delta_sheet(args: Tuple[Path, float, str]) -> np.ndarray:
    """
    Process a single delta sheet and return its contribution to the kappa map.

    Args:
        args: A tuple containing:
            - data_path: Path to the delta sheet FITS file.
            - wlen_int: Precomputed weak lensing integral value.
            - dtype: Data type for reading the map.

    Returns:
        The delta contribution as a NumPy array.
    """
    data_path, wlen_int, dtype = args
    logging.info(f"Processing {data_path.name} with wlen_int={wlen_int}")
    delta = hp.read_map(data_path).astype(dtype)
    delta_contribution = delta * wlen_int
    return delta_contribution

class WeakLensingHelper:
    @staticmethod
    def wlen_chi_kappa(chi: float, cosmo: FlatLambdaCDM, zs: float) -> float:
        """Compute the weight function for weak lensing convergence."""
        chis = cosmo.comoving_distance(zs).value # Mpc
        H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
        z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
        dchi = (1 - chi / chis).clip(0)
        return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi # 1/Mpc
    
    @staticmethod
    def index_to_chi(index: int, cosmo: FlatLambdaCDM) -> Tuple[float, float]:
        """Convert an index to a comoving distance."""
        a1, a2 = 0.01 * index, 0.01 * (index + 1)
        z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value # [Mpc]
        return chi1, chi2
    
    @staticmethod
    def wlen_integral(chi1: float, chi2: float, wlen: callable, cosmology, zs: float) -> float:
        """Compute the weak lensing integral."""
        chi = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
        dchi = chi1 - chi2
        return wlen(chi, cosmology, zs) * dchi

class KappaConstructor:
    def __init__(self, datadir, nside=8192, zs_list=None, overwrite=False):
        self.datadir = Path(datadir)
        self.seed = InfoExtractor.extract_info_from_path(self.datadir)['seed']
        self.zs_list = zs_list or [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self._initialize_directories()

        self.dtype = 'float32'
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        # Precompute chi values for all delta sheets
        self.sheet_indices = [
            int(f.stem.split("-")[-1]) for f in self.sheet_files
        ]
        self.chi_pairs = [
            WeakLensingHelper.index_to_chi(i, self.cosmo) for i in self.sheet_indices
        ]
        
    def _initialize_directories(self):
        self.outputdir = self.datadir / "kappa"
        self.outputdir.mkdir(exist_ok=True)

        self.massdir = self.datadir / "mass_sheets"
        self.sheet_files = sorted(self.massdir.glob("delta-sheet-*.fits"))

    def compute_kappa(self):
        for zs in self.zs_list:
            logging.info(f"Starting computation for zs={zs}.")
            kappa_file = self.outputdir / f"kappa_zs{zs}_s{self.seed}.fits"
            if kappa_file.exists() and not self.overwrite:
                logging.info(f"Kappa for zs={zs} exists. Skipping.")
                continue    

            # Precompute wlen_integral values for each sheet
            wlen_integrals = [
                WeakLensingHelper.wlen_integral(
                    chi1, chi2, WeakLensingHelper.wlen_chi_kappa, self.cosmo, zs
                )
                for chi1, chi2 in self.chi_pairs
            ]

            kappa = self._compute_kappa(zs, wlen_integrals)
            hp.write_map(str(kappa_file), kappa, dtype=np.float32)
            logging.info(f"Kappa saved to {kappa_file.name}.")

    def _compute_kappa(self, zs: float, wlen_integrals: List[float]) -> np.ndarray:
        """Compute the global kappa map using multiprocessing."""
        args_list = [
            (data_path, wlen_int, self.dtype)
            for data_path, wlen_int in zip(self.sheet_files, wlen_integrals)
        ]

        kappa_map = np.zeros(self.npix, dtype=self.dtype)

        with Pool() as pool:
            for delta_contrib in pool.imap_unordered(process_delta_sheet, args_list):
                kappa_map += delta_contrib

        return kappa_map
        
if __name__ == "__main__":
    from src.utils import parse_arguments, load_config, filter_config, setup_logging
    from pathlib import Path

    setup_logging()
    args = parse_arguments()
    config = load_config(args.config_file)

    kc_config = filter_config(config, KappaConstructor)
    kc = KappaConstructor(args.datadir, **kc_config)
    kc.compute_kappa()