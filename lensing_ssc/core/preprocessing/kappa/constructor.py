import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from multiprocessing import Pool

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from lensing_ssc.utils.extractors import InfoExtractor
from lensing_ssc.core.preprocessing.utils.weight_functions import (
    compute_weight_function, compute_wlen_integral, index_to_chi_pair
)


def process_delta_sheet(args: Tuple[Path, float, str]) -> np.ndarray:
    """
    Process a single delta sheet FITS file and return its contribution
    to the kappa map.
    """
    data_path, wlen_int, dtype = args
    logging.info(f"Processing {data_path.name} with wlen_int={wlen_int}")
    try:
        delta_map = hp.read_map(data_path)
    except OSError as e:
        logging.error(f"Failed to read {data_path.name}: {e}")
        raise

    delta_contribution = delta_map.astype(dtype) * wlen_int
    return delta_contribution


class KappaConstructor:
    """
    A class for constructing kappa (convergence) maps from delta sheets.
    """

    def __init__(
        self,
        datadir: str,
        nside: int = 8192,
        zs_list: Optional[List[float]] = None,
        overwrite: bool = False,
        num_workers: Optional[int] = None
    ) -> None:
        """Initialize the KappaConstructor."""
        self.datadir = Path(datadir)
        self.seed = self._extract_seed()
        self.zs_list = zs_list or [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self.dtype = "float32"
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        self.outputdir = self.datadir / "kappa"
        self.outputdir.mkdir(exist_ok=True)

        self.massdir = self.datadir / "mass_sheets"
        self.sheet_files = sorted(self.massdir.glob("delta-sheet-*.fits"))
        self.sheet_indices = [int(f.stem.split("-")[-1]) for f in self.sheet_files]
        self.chi_pairs = [
            index_to_chi_pair(i, self.cosmo)
            for i in self.sheet_indices
        ]

    def _extract_seed(self) -> str:
        """Extract the random seed from the data directory."""
        extracted_info = InfoExtractor.extract_info_from_path(self.datadir)
        return extracted_info.get("seed", "unknown")

    def compute_kappa(self) -> None:
        """Compute and save the kappa map for each source redshift in self.zs_list."""
        for zs in self.zs_list:
            logging.info(f"Starting kappa computation for zs={zs}.")
            kappa_file = self.outputdir / f"kappa_zs{zs}_s{self.seed}.fits"
            if kappa_file.exists() and not self.overwrite:
                logging.info(f"Kappa for zs={zs} already exists. Skipping.")
                continue

            wlen_integrals = self._precompute_wlen_integrals(zs)
            kappa_map = self._compute_kappa_map(wlen_integrals)
            hp.write_map(str(kappa_file), kappa_map, dtype=np.float32)
            logging.info(f"Kappa saved to {kappa_file.name}.")

    def _precompute_wlen_integrals(self, zs: float) -> List[float]:
        """Precompute weak lensing integrals for all mass sheets at a given source redshift."""
        return [
            compute_wlen_integral(
                chi1, chi2,
                compute_weight_function,
                self.cosmo,
                zs
            )
            for chi1, chi2 in self.chi_pairs
        ]

    def _compute_kappa_map(self, wlen_integrals: List[float]) -> np.ndarray:
        """Compute the global kappa map by summing the contributions from each delta sheet."""
        kappa_map = np.zeros(self.npix, dtype=self.dtype)
        args_list = [
            (data_path, wlen_int, self.dtype)
            for data_path, wlen_int in zip(self.sheet_files, wlen_integrals)
        ]
        
        with Pool(processes=self.num_workers) as pool:
            for delta_contrib in pool.imap_unordered(process_delta_sheet, args_list):
                kappa_map += delta_contrib
        return kappa_map