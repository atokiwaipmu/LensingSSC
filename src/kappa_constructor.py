
import os
import glob
import logging
import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import multiprocessing as mp
from functools import partial

class KappaConstructor:
    def __init__(self, datadir, zs_list = [0.5, 1.0, 2.0, 3.0], overwrite=False):
        self.datadir = datadir
        self.zs_list = zs_list
        self.overwrite = overwrite
        self._initialization()

        self.cosmo = FlatLambdaCDM(H0=0.6774 * 100, Om0=0.309)
        
    def _initialization(self):
        self.outputdir = os.path.join(self.datadir, "kappa")
        os.makedirs(self.outputdir, exist_ok=True)
        self.massdir = os.path.join(self.datadir, "mass_sheets")
        self.sheet_files = sorted(glob.glob(os.path.join(self.massdir, "delta-sheet-*.fits")))

    def compute_kappa(self):
        for zs in self.zs_list:
            logging.info(f"Starting the computation of weak lensing convergence maps for zs={zs}.")
            kappa_file = os.path.join(self.outputdir, f"kappa_zs{zs}.fits")
            if os.path.exists(kappa_file) and not self.overwrite:
                logging.info(f"Kappa file for zs={zs} already exists. Skipping...")
                continue
            kappa = self._compute_kappa(zs)
            hp.write_map(kappa_file, kappa, dtype=np.float32)
            logging.info(f"Kappa file for zs={zs} saved to {kappa_file}")

    def _compute_kappa(self, zs):
        # Initialize SheetMapper and create maps
        mapper = SheetMapper()
        mapper.new_map("kappa")

        # Create a partial function with fixed arguments for use with multiprocessing
        process_partial = partial(self._process_delta_sheet, zs=zs, mapper=mapper)
        
        # Use multiprocessing Pool to parallelize the work
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(process_partial, self.sheet_files)

        # Reduce the results across all processes
        global_kappa = self.reduce_maps(results)

        return global_kappa

    def _process_delta_sheet(self, data_path, zs, mapper):
        logging.info(f"Processing delta sheet index {i}")
        delta = hp.read_map(data_path)
        i = int(data_path.split("-")[-1].split(".")[0])
        chi1, chi2 = WeakLensingHelper.index_to_chi(i, self.cosmo)
        mapper.add_sheet_to_map("kappa", delta.astype('float32'), WeakLensingHelper.wlen_chi_kappa, chi1, chi2, self.cosmo, zs)
        return mapper.maps["kappa"]
    
    def reduce_maps(results):
        global_kappa = np.zeros_like(results[0])
        for local_kappa in results:
            global_kappa += local_kappa
        return global_kappa

class SheetMapper:
    """Handles operations related to sheet mapping for cosmological data visualization."""

    def __init__(self, nside=8192):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.maps = {}

    def new_map(self, map_name, dtype='float32'):
        """Create a new map with the given name and data type."""
        self.maps[map_name] = np.zeros(self.npix, dtype=dtype)

    def add_sheet_to_map(self, map_name, sheet, wlen, chi1, chi2, cosmology, zs):
        """Add a sheet to the map using a weak lensing integral."""
        chi =  3/4 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)  # [Mpc]
        dchi = chi1 - chi2 # [Mpc]
        wlen_integral = wlen(chi, cosmology, zs) * dchi  # [1]
        self.maps[map_name] += wlen_integral * sheet  

class WeakLensingHelper:
    @staticmethod
    def wlen_chi_kappa(chi, cosmo, zs):
        """Compute the weight function for weak lensing convergence."""
        chis = cosmo.comoving_distance(zs).value # Mpc
        H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
        z = cosmology.z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
        dchi = (1 - chi / chis).clip(0)
        return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi # 1/Mpc
    
    @staticmethod
    def index_to_chi(index, cosmo):
        """Convert an index to a comoving distance."""
        a1, a2 = 0.01 * index, 0.01 * (index + 1)
        z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * cosmo.h
        return chi1, chi2