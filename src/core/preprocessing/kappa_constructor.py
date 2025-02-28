import logging
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import constants as const, units as u

from utils.info_extractor import InfoExtractor


def process_delta_sheet(args: Tuple[Path, float, str]) -> np.ndarray:
    """
    Process a single delta sheet FITS file and return its contribution
    to the kappa map.

    Parameters
    ----------
    args : Tuple[Path, float, str]
        A tuple containing:
          - data_path: Path to the delta sheet FITS file.
          - wlen_int: Precomputed weak lensing integral for the sheet.
          - dtype: Data type for reading the map.

    Returns
    -------
    np.ndarray
        The delta contribution array for this sheet.
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


class WeakLensingHelper:
    """
    A helper class for weak lensing related calculations.
    """

    @staticmethod
    def compute_weight_function(chi: float,
                                cosmo: FlatLambdaCDM,
                                zs: float) -> float:
        """
        Compute the weight function for weak lensing convergence.

        Parameters
        ----------
        chi : float
            Comoving distance for the lens plane in Mpc.
        cosmo : FlatLambdaCDM
            Cosmology object.
        zs : float
            Source redshift.

        Returns
        -------
        float
            Weight function value at comoving distance chi.
        """
        chis = cosmo.comoving_distance(zs).value  # Mpc
        # Convert H0 from km/s/Mpc to 1/Mpc
        H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
        z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
        dchi = (1.0 - chi / chis).clip(0.0)

        return 3.0 / 2.0 * cosmo.Om0 * (H0 ** 2) * (1.0 + z) * chi * dchi

    @staticmethod
    def index_to_chi_pair(index: int,
                          cosmo: FlatLambdaCDM) -> Tuple[float, float]:
        """
        Convert an integer index (representing a mass sheet) to a pair of
        comoving distances (chi1, chi2).

        Parameters
        ----------
        index : int
            Index representing the mass sheet.
        cosmo : FlatLambdaCDM
            Cosmology object.

        Returns
        -------
        Tuple[float, float]
            The lower and upper comoving distances in Mpc for the sheet.
        """
        a1, a2 = 0.01 * index, 0.01 * (index + 1)
        z1, z2 = (1.0 / a1 - 1.0), (1.0 / a2 - 1.0)
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value  # Mpc
        return chi1, chi2

    @staticmethod
    def compute_wlen_integral(chi1: float,
                              chi2: float,
                              wlen_func: callable,
                              cosmo: FlatLambdaCDM,
                              zs: float) -> float:
        """
        Compute the weak lensing integral for a single sheet.

        Parameters
        ----------
        chi1 : float
            Lower comoving distance bound in Mpc.
        chi2 : float
            Upper comoving distance bound in Mpc.
        wlen_func : callable
            Weight function for weak lensing (e.g., compute_weight_function).
        cosmo : FlatLambdaCDM
            Cosmology object.
        zs : float
            Source redshift.

        Returns
        -------
        float
            The integrated weak lensing contribution from chi1 to chi2.
        """
        # Approximate midpoint distance for the integral
        chi_mid = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
        dchi = chi1 - chi2
        return wlen_func(chi_mid, cosmo, zs) * dchi


class KappaConstructor:
    """
    A class for constructing kappa (convergence) maps from delta sheets.
    """

    def __init__(self,
                 datadir: str,
                 nside: int = 8192,
                 zs_list: Optional[List[float]] = None,
                 overwrite: bool = False) -> None:
        """
        Initialize the KappaConstructor.

        Parameters
        ----------
        datadir : str
            Path to the main data directory.
        nside : int, optional
            Healpy NSIDE parameter, by default 8192.
        zs_list : List[float], optional
            List of source redshifts to process, by default [0.5, 1.0, 1.5, 2.0, 2.5].
        overwrite : bool, optional
            Whether to overwrite existing files, by default False.
        """
        self.datadir = Path(datadir)
        self.seed = self._extract_seed()
        self.zs_list = zs_list or [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self.dtype = "float32"
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        self.outputdir = self.datadir / "kappa"
        self.outputdir.mkdir(exist_ok=True)

        self.massdir = self.datadir / "mass_sheets"
        self.sheet_files = sorted(self.massdir.glob("delta-sheet-*.fits"))

        # Precompute the comoving distance bounds for each mass sheet
        self.sheet_indices = [int(f.stem.split("-")[-1]) for f in self.sheet_files]
        self.chi_pairs = [
            WeakLensingHelper.index_to_chi_pair(i, self.cosmo)
            for i in self.sheet_indices
        ]

    def _extract_seed(self) -> str:
        """
        Extract the random seed from the data directory using InfoExtractor.

        Returns
        -------
        str
            The seed extracted from the directory info.
        """
        extracted_info = InfoExtractor.extract_info_from_path(self.datadir)
        return extracted_info["seed"]

    def compute_kappa(self) -> None:
        """
        Compute and save the kappa map for each source redshift in self.zs_list.
        """
        for zs in self.zs_list:
            logging.info(f"Starting kappa computation for zs={zs}.")

            kappa_file = self.outputdir / f"kappa_zs{zs}_s{self.seed}.fits"
            if kappa_file.exists() and not self.overwrite:
                logging.info(f"Kappa for zs={zs} already exists. Skipping.")
                continue

            # Precompute the weak lensing integrals for all sheets at this source redshift
            wlen_integrals = self._precompute_wlen_integrals(zs)

            # Compute the global kappa map
            kappa_map = self._compute_kappa_map(wlen_integrals)

            # Save to disk
            hp.write_map(str(kappa_file), kappa_map, dtype=np.float32)
            logging.info(f"Kappa saved to {kappa_file.name}.")

    def _precompute_wlen_integrals(self, zs: float) -> List[float]:
        """
        Precompute weak lensing integrals for all sheets at a given source redshift.

        Parameters
        ----------
        zs : float
            The source redshift.

        Returns
        -------
        List[float]
            List of weak lensing integrals, one for each mass sheet.
        """
        return [
            WeakLensingHelper.compute_wlen_integral(
                chi1, chi2,
                WeakLensingHelper.compute_weight_function,
                self.cosmo,
                zs
            )
            for chi1, chi2 in self.chi_pairs
        ]

    def _compute_kappa_map(self, wlen_integrals: List[float]) -> np.ndarray:
        """
        Compute the global kappa map by summing contributions from each delta sheet.

        Parameters
        ----------
        wlen_integrals : List[float]
            Precomputed weak lensing integrals for each sheet.

        Returns
        -------
        np.ndarray
            The aggregated kappa map.
        """
        kappa_map = np.zeros(self.npix, dtype=self.dtype)
        args_list = [
            (data_path, wlen_int, self.dtype)
            for data_path, wlen_int in zip(self.sheet_files, wlen_integrals)
        ]

        with Pool() as pool:
            for delta_contrib in pool.imap_unordered(process_delta_sheet, args_list):
                kappa_map += delta_contrib

        return kappa_map


if __name__ == "__main__":
    from utils.utils import parse_arguments, load_config, filter_config, setup_logging

    setup_logging()
    args = parse_arguments()
    config = load_config(args.config_file)

    # Filter out keyword arguments that match the KappaConstructor signature
    kc_config = filter_config(config, KappaConstructor)
    kc = KappaConstructor(args.datadir, **kc_config)
    kc.compute_kappa()