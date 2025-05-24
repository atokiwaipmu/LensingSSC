# lensing_ssc/core/preprocessing_utils.py
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import pandas as pd
import numpy as np
import healpy as hp
from nbodykit.lab import BigFileCatalog
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const, units as u
from astropy.cosmology import z_at_value
from multiprocessing import Pool
import re


class IndicesFinder:
    """Class to find and save processing indices for mass sheets data."""

    def __init__(self, datadir: Path, seed: int, extra_index: int = 100) -> None:
        """
        Initialize the IndicesFinder.

        Args:
            datadir (Path): Directory containing the mass sheets data.
            seed (int): Seed identifier for naming the indices CSV file.
            extra_index (int, optional): Extra index for processing. Defaults to 100.
        """
        self.datadir = datadir
        self.initial_seed = seed  # original seed provided
        self.extra_index = extra_index

        self.msheets = BigFileCatalog(str(self.datadir / "usmesh"), dataset="HEALPIX/")
        # Use the seed from the dataset if available
        self.seed = self.msheets.attrs.get('seed', [seed])[0]
        self.aemit_index_offset = self.msheets.attrs['aemitIndex.offset']
        self.aemit_index_edges = self.msheets.attrs['aemitIndex.edges']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]

        self.save_path = self.datadir / f"preproc_s{self.seed}_indices.csv"

    def find_indices(self, i_start: int = 20, i_end: int = 100) -> None:
        """
        Find and save the start and end indices for each mass sheet.

        Args:
            i_start (int, optional): Starting sheet index. Defaults to 20.
            i_end (int, optional): Ending sheet index. Defaults to 100.
        """
        indices: List[dict] = []
        prev_end: Optional[int] = None

        for i in range(i_start, i_end):
            if self.is_sheet_empty(i):
                logging.info(f"Sheet {i} is empty. Skipping...")
                continue

            start, end = self._find_index(i, start=prev_end)
            prev_end = end
            indices.append({"sheet": i, "start": start, "end": end})

        if indices:
            indices_df = pd.DataFrame(indices)
            indices_df.to_csv(self.save_path, index=False)
            logging.info(f"Indices saved to {self.save_path}")
        else:
            logging.warning("No indices were found to save.")

    def is_sheet_empty(self, sheet: int) -> bool:
        """
        Check if the given sheet is empty.

        Args:
            sheet (int): Sheet index to check.

        Returns:
            bool: True if the sheet is empty, otherwise False.
        """
        return self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]

    def _find_index(self, sheet: int, start: Optional[int] = None) -> Tuple[int, int]:
        """
        Determine the start and end indices for processing a mass sheet.

        Args:
            sheet (int): Sheet index.
            start (Optional[int], optional): Previous end index. Defaults to None.

        Returns:
            Tuple[int, int]: Tuple containing start and end indices.
        """
        if start is not None:
            start_index = start
            end_index = self.aemit_index_offset[sheet + 2]
        else:
            start_index = self.aemit_index_offset[sheet + 1]
            end_index = self.aemit_index_offset[sheet + 2]

            if self.extra_index:
                search_start = min(start_index + self.extra_index, end_index)
                aemit_slice = self.msheets['Aemit'][start_index:search_start].compute()
                diff = np.diff(aemit_slice)
                change_indices = np.where(diff == self.a_interval)[0]
                if change_indices.size > 0:
                    delta = change_indices[0]
                    logging.info(f"Aemit start changed from {start_index} to {start_index + delta}")
                    start_index += delta

        if start_index == end_index:
            return start_index, end_index

        logging.info(f"Determined indices - start: {start_index}, end: {end_index}")

        if self.extra_index:
            search_end = max(end_index - self.extra_index, start_index)
            aemit_slice_end = self.msheets['Aemit'][search_end:end_index].compute()
            diff_end = np.round(np.diff(aemit_slice_end), 2)
            change_indices_end = np.where(diff_end == self.a_interval)[0]
            if change_indices_end.size > 0:
                delta = change_indices_end[0]
                logging.info(f"Aemit end changed from {end_index} to {end_index - delta}")
                end_index -= delta

        return start_index, end_index


class MassSheetProcessor:
    """Processes mass sheets for cosmological data visualization and analysis."""

    def __init__(self, datadir: Path, overwrite: bool = False) -> None:
        """
        Initialize the MassSheetProcessor.

        Args:
            datadir (Path): Directory containing the mass sheets data.
            overwrite (bool, optional): Overwrite existing processed files. Defaults to False.
        """
        self.datadir = datadir
        self.overwrite = overwrite

        self.msheets = BigFileCatalog(str(self.datadir / "usmesh"), dataset="HEALPIX/")
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)

        self._initialize_output_directory()
        self._initialize_attributes()
        self.indices_df = self._load_precomputed_indices()

    def _initialize_output_directory(self) -> None:
        """Create the output directory for processed mass sheets."""
        self.output_dir = self.datadir / "mass_sheets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_attributes(self) -> None:
        """Initialize catalog attributes."""
        attrs = self.msheets.attrs
        self.aemit_index_edges = attrs['aemitIndex.edges']
        self.aemit_index_offset = attrs['aemitIndex.offset']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]
        self.npix = attrs['healpix.npix'][0]
        self.box_size = attrs['BoxSize'][0]
        self.m_cdm = attrs['MassTable'][1]
        self.nc = attrs['NC'][0]
        self.rhobar = self.m_cdm * (self.nc / self.box_size) ** 3
        
        # Try to get seed from attributes, if not found, try to extract from dataset name
        try:
            self.seed = attrs['seed'][0]
        except KeyError:
            # Extract seed from dataset name (assuming format contains 's{number}')
            dataset_name = str(self.datadir)
            seed_match = re.search(r's(\d+)', dataset_name)
            if seed_match:
                self.seed = int(seed_match.group(1))
            else:
                # If no seed found, use a default value
                self.seed = 0
                logging.warning("No seed found in attributes or dataset name. Using default seed=0")

    def _load_precomputed_indices(self) -> pd.DataFrame:
        """
        Load precomputed indices from CSV. If not found, run IndicesFinder.

        Returns:
            pd.DataFrame: DataFrame with columns 'sheet', 'start', and 'end'.
        """
        csv_path = self.datadir / f"preproc_s{self.seed}_indices.csv"
        if not csv_path.is_file():
            logging.error(f"Precomputed indices file not found: {csv_path}. Running IndicesFinder.")
            finder = IndicesFinder(self.datadir, self.seed)
            finder.find_indices()
        df = pd.read_csv(csv_path)
        if not {'sheet', 'start', 'end'}.issubset(df.columns):
            logging.error("CSV file is missing required columns: 'sheet', 'start', 'end'.")
            raise ValueError("CSV file must contain columns: 'sheet', 'start', 'end'")
        return df

    def preprocess(self) -> None:
        """
        Preprocess mass sheets using precomputed indices and save the processed maps.
        """
        for _, row in self.indices_df.iterrows():
            sheet = int(row['sheet'])
            start = int(row['start'])
            end = int(row['end'])

            if self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]:
                logging.info(f"Sheet {sheet} is empty. Skipping...")
                continue

            save_path = self.output_dir / f"delta-sheet-{sheet:02d}.fits"
            if save_path.exists() and not self.overwrite:
                logging.info(f"File {save_path} exists and overwrite is False. Skipping...")
                continue

            delta = self._get_mass_sheet(sheet, start, end)
            hp.write_map(str(save_path), delta, nest=True, dtype=np.float32)
            logging.info(f"Saved processed sheet {sheet} to {save_path}")

    def _get_mass_sheet(self, sheet: int, start: int, end: int) -> np.ndarray:
        """
        Process a mass sheet to compute the density contrast map.

        Args:
            sheet (int): Sheet index.
            start (int): Start index for data extraction.
            end (int): End index for data extraction.

        Returns:
            np.ndarray: Computed density contrast map.
        """
        logging.info(f"Processing sheet {sheet}: reading IDs and Mass values.")
        pid = self.msheets['ID'][start:end].compute()
        mass = self.msheets['Mass'][start:end].compute()
        ipix = pid % self.npix
        map_slice = np.bincount(ipix, weights=mass, minlength=self.npix)

        a1, a2 = self.aemit_index_edges[sheet:sheet + 2]
        z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
        chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
        volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * self.npix)
        delta = map_slice / (volume_diff * self.rhobar) - 1.0

        return delta


# Functions from legacy weight_functions.py
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
    chis = cosmo.comoving_distance(zs).value  # in Mpc
    # Convert H0 from km/s/Mpc to 1/Mpc
    # H0_inv_mpc = cosmo.H0.to(u.km / u.s / u.Mpc).value / (const.c.to(u.km / u.s).value)
    # The original H0 conversion was a bit obscure. Let's use astropy units more directly.
    # However, the formula 1.5 * Om0 * (H0/c)^2 * (1+z) * chi * (1-chi/chis) * c^2 is also common.
    # The original formula: 1.5 * cosmo.Om0 * (H0 ** 2) * (1.0 + z) * chi * dchi
    # where H0 was in 1/Mpc. H0/c in natural units. Let's stick to original for now if it worked.
    H0_val = 100 * cosmo.h # H0 in km/s/Mpc
    # Convert H0 to 1/Mpc by dividing by c in km/s
    H0_inv_mpc_squared = (H0_val / const.c.to(u.km/u.s).value)**2

    # Original had H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5), which is H0 / (c_in_km_s)
    # This is H0_km_s_mpc / c_km_s. So (H0/c)^2 is what is needed.

    z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value # z at comoving distance chi
    dchi_factor = np.clip(1.0 - (chi / chis), 0.0, None) # Avoid negative if chi > chis due to numerical precision
    # Weight function definition: (3/2) * Om0 * (H0/c)^2 * (1+z) * chi * (chis-chi)/chis * c^2 (if H0/c is 1/length)
    # Or simply: 1.5 * Om0 * (H0_val / c_kms_val)^2 * (1+z) * chi * (1 - chi/chis)
    # The formula used seems to be W = 1.5 * Om_m * (H_0/c)^2 * (1+z_lens) * chi_lens * (chi_source - chi_lens) / chi_source
    # Here, H0 is in units of 1/Mpc, chi is in Mpc.
    # The constant 1.5 * Om0 * (H0 [1/Mpc])^2 is prefactor for kappa integral.
    # Let's verify the units and formula. Typical lensing kernel is prop. to (1+z) * chi * (chis-chi)/chis.
    # The prefactor 1.5 * Om0 * (H0/c)^2 combines with this. H0/c has units 1/distance.
    # If H0 is passed as just 'H0' from astropy, it has units km/s/Mpc.
    # (H0 [km/s/Mpc] / c [km/s])^2 gives (1/Mpc)^2.

    # Sticking to the original form as it was likely tested:
    # H0_orig = 100 * cosmo.h / (const.c.cgs.value / 1e5) # This is H0 in 1/Mpc (cgs.value/1e5 is c in km/s)
    # This is equivalent to H0_val / const.c.to(u.km/u.s).value

    H0_for_formula = cosmo.H0.value / const.c.to(u.km/u.s).value # H0 in 1/Mpc

    return 1.5 * cosmo.Om0 * (H0_for_formula ** 2) * (1.0 + z) * chi * dchi_factor

def compute_wlen_integral(chi1: float,
                          chi2: float,
                          wlen_func: Callable[[float, FlatLambdaCDM, float], float],
                          cosmo: FlatLambdaCDM,
                          zs: float) -> float:
    """
    Compute the weak lensing integral for a single mass sheet.
    Uses a midpoint approximation for the integral: w_L(chi_mid) * (chi1 - chi2).

    Parameters
    ----------
    chi1 : float
        Lower comoving distance bound in Mpc (larger value, closer to observer).
    chi2 : float
        Upper comoving distance bound in Mpc (smaller value, further from observer).
    wlen_func : callable
        Weight function (e.g., compute_weight_function).
    cosmo : FlatLambdaCDM
        Cosmology object.
    zs : float
        Source redshift.

    Returns
    -------
    float
        The integrated weak lensing contribution from chi2 to chi1.
    """
    # The original code had chi_mid = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
    # This is the centroid for a volume element that scales as chi^2 dchi. This seems correct.
    if chi1 == chi2: # Avoid division by zero if shell is infinitesimally thin
        return 0.0
    if chi1 < chi2: # Ensure chi1 > chi2 as dchi = chi1-chi2 is positive thickness
        chi1, chi2 = chi2, chi1
        
    chi_mid = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
    delta_chi = chi1 - chi2 # Thickness of the shell
    return wlen_func(chi_mid, cosmo, zs) * delta_chi

def index_to_chi_pair(index: int, cosmo: FlatLambdaCDM, a_edges: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Convert a mass sheet index to a pair of comoving distance bounds.
    If a_edges is provided, it uses those scale factor edges.
    Otherwise, assumes a default a_i = 0.01 * index.

    Parameters
    ----------
    index : int
        Index representing the mass sheet (corresponds to an interval in a_edges).
    cosmo : FlatLambdaCDM
        Cosmology object.
    a_edges: np.ndarray, optional
        Array of scale factor bin edges. If given, index refers to the i-th interval [a_edges[i], a_edges[i+1]].

    Returns
    -------
    Tuple[float, float]
        Comoving distances (chi_inner, chi_outer) in Mpc, where chi_inner corresponds to larger 'a' (closer to z=0).
        Order is (chi(a_i+1), chi(a_i)) -> (chi_closer, chi_further_away)
    """
    if a_edges is not None:
        if index + 1 >= len(a_edges):
            raise ValueError(f"Index {index} out of bounds for a_edges with length {len(a_edges)}")
        a_outer = a_edges[index]      # smaller a, larger z, further distance
        a_inner = a_edges[index+1]  # larger a, smaller z, closer distance
    else:
        # Original default logic if a_edges not from usmesh attributes
        a_outer = 0.01 * index       # This assumes a_i = 0.01 * i, so a_0=0, a_1=0.01 etc.
                                     # If index is for sheet between a_i and a_i+1
        a_inner = 0.01 * (index + 1) # This means index=0 is a=0 to a=0.01
    
    # Ensure a_outer < a_inner for correct z ordering (z_outer > z_inner)
    if a_outer == 0.0: # Avoid z=inf for a=0, start from a very small a if index implies from a=0
        z_outer = 1e6 # Effectively infinity for comoving distance calculation
    else:
        z_outer = (1.0 / a_outer) - 1.0
    
    z_inner = (1.0 / a_inner) - 1.0

    # Comoving distance is larger for larger z (smaller a)
    chi_further = cosmo.comoving_distance(z_outer).value
    chi_closer = cosmo.comoving_distance(z_inner).value
    
    # Return (closer distance, further distance) to match (chi1, chi2) where chi1 > chi2 in integral
    return chi_closer, chi_further


def process_delta_sheet(args: Tuple[Path, float, np.dtype]) -> Optional[np.ndarray]: # Added np.dtype hint
    """
    Process a single delta sheet FITS file and return its contribution
    to the kappa map.
    """
    data_path, wlen_int, dtype_str = args # dtype_str is string like 'float32'
    logging.info(f"Processing {data_path.name} with wlen_int={wlen_int:.4e}")
    try:
        # Ensure map is read in the native NEST ordering if that's how it's stored
        delta_map = hp.read_map(str(data_path), nest=None) # Use str() for Path object
    except OSError as e:
        logging.error(f"Failed to read {data_path.name}: {e}")
        return None # Return None or raise to be handled by caller
    except Exception as e: # Catch other potential errors during read_map
        logging.error(f"Unexpected error reading {data_path.name}: {e}")
        return None

    # Ensure delta_map is 1D array for hp.nside2npix consistency if needed for checks
    # npix_map = hp.get_map_size(delta_map)
    # if npix_map == 0: # Indicates not a valid healpix map or empty
    #     logging.warning(f"Map {data_path.name} is empty or not a valid Healpix map.")
    #     return np.zeros(hp.nside2npix(self.nside), dtype=dtype_str) # Use constructor's nside for consistency
        
    delta_contribution = delta_map.astype(np.dtype(dtype_str)) * wlen_int # Use np.dtype() to convert string
    return delta_contribution


class KappaConstructor:
    """
    A class for constructing kappa (convergence) maps from delta (mass) sheets.
    Assumes mass sheets are Healpix maps named like 'delta-sheet-{index:02d}.fits'.
    """

    def __init__(
        self,
        mass_sheet_dir: Path, # Changed from datadir to be more specific
        output_dir: Path,
        usmesh_attrs: dict, # Pass attributes from usmesh (e.g., from MassSheetProcessor.msheets.attrs)
        nside: int = 8192,
        zs_list: Optional[List[float]] = None,
        overwrite: bool = False,
        num_workers: Optional[int] = None,
        cosmo_params: Optional[dict] = None # For FlatLambdaCDM
    ) -> None:
        """Initialize the KappaConstructor."""
        self.mass_sheet_dir = Path(mass_sheet_dir)
        self.output_dir = Path(output_dir)
        self.usmesh_attrs = usmesh_attrs
        self.seed = str(self.usmesh_attrs.get("seed", ["unknown"])[0]) # Extract seed from usmesh_attrs
        
        self.zs_list = zs_list if zs_list is not None else [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.num_workers = num_workers if num_workers is not None else Pool()._processes # Default to cpu_count
        
        default_cosmo = {"H0": 67.74, "Om0": 0.309}
        if cosmo_params:
            default_cosmo.update(cosmo_params)
        self.cosmo = FlatLambdaCDM(H0=default_cosmo["H0"], Om0=default_cosmo["Om0"])
        
        self.dtype_str = "float32" # Store as string for process_delta_sheet
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.mass_sheet_dir.exists():
            raise FileNotFoundError(f"Mass sheet directory not found: {self.mass_sheet_dir}")

        self.sheet_files = sorted(self.mass_sheet_dir.glob("delta-sheet-*.fits"))
        if not self.sheet_files:
            logging.warning(f"No delta-sheet-*.fits files found in {self.mass_sheet_dir}")
            self.sheet_indices = []
            self.a_edges = np.array([]) # from usmesh_attrs['aemitIndex.edges']
        else:
            self.sheet_indices = [int(f.stem.split("-")[-1]) for f in self.sheet_files]
        
        # Get a_edges from usmesh attributes, critical for index_to_chi_pair
        self.a_edges = self.usmesh_attrs.get('aemitIndex.edges')
        if self.a_edges is None:
            logging.warning("'aemitIndex.edges' not found in usmesh_attrs. Falling back to default a_i = 0.01*i for chi calculation.")
            # This fallback might be inaccurate if sheets were not generated with this assumption.

        # Precompute chi_pairs using a_edges if available
        self.chi_pairs = []
        for i in self.sheet_indices:
            try:
                # Pass self.a_edges to index_to_chi_pair
                # The 'index' for delta-sheet-{index}.fits corresponds to the interval a_edges[index] to a_edges[index+1]
                self.chi_pairs.append(index_to_chi_pair(i, self.cosmo, self.a_edges))
            except ValueError as e:
                logging.error(f"Error getting chi_pair for sheet index {i}: {e}")
                # Decide how to handle this: skip sheet, raise error, or use default?
                # For now, let's log and potentially skip if chi_pair is problematic.
                # A robust solution would ensure sheet_indices align with a_edges intervals.

    def compute_all_kappas(self) -> None:
        """Compute and save the kappa map for each source redshift in self.zs_list."""
        if not self.sheet_files or not self.chi_pairs:
            logging.error("No mass sheets or chi_pairs available to compute kappa maps. Aborting.")
            return
            
        for zs in self.zs_list:
            logging.info(f"Starting kappa computation for zs={zs}.")
            kappa_file_name = f"kappa_zs{zs}_s{self.seed}_nside{self.nside}.fits" 
            kappa_file = self.output_dir / kappa_file_name
            
            if kappa_file.exists() and not self.overwrite:
                logging.info(f"Kappa map {kappa_file.name} for zs={zs} already exists. Skipping.")
                continue

            wlen_integrals = self._precompute_wlen_integrals(zs)
            if len(wlen_integrals) != len(self.sheet_files):
                logging.error(f"Mismatch between number of wlen_integrals ({len(wlen_integrals)}) and sheet_files ({len(self.sheet_files)}) for zs={zs}. Skipping.")
                continue
                
            kappa_map = self._compute_kappa_map_for_zs(wlen_integrals)
            if kappa_map is None:
                logging.error(f"Failed to compute kappa map for zs={zs}. Skipping save.")
                continue
            
            hp.write_map(str(kappa_file), kappa_map, dtype=np.dtype(self.dtype_str), nest=None, overwrite=self.overwrite)
            logging.info(f"Kappa map saved to {kappa_file.name}.")

    def _precompute_wlen_integrals(self, zs: float) -> List[float]:
        """Precompute weak lensing integrals for all mass sheets at a given source redshift."""
        integrals = []
        for chi_pair_for_sheet in self.chi_pairs:
            chi_closer, chi_further = chi_pair_for_sheet # chi_closer > chi_further
            # Ensure chi_closer (from a_inner) and chi_further (from a_outer) are correctly ordered for integral
            integrals.append(
                compute_wlen_integral(
                    chi_closer, chi_further, # chi1 > chi2 for integral
                    compute_weight_function,
                    self.cosmo,
                    zs
                )
            )
        return integrals

    def _compute_kappa_map_for_zs(self, wlen_integrals: List[float]) -> Optional[np.ndarray]:
        """Compute the global kappa map by summing the contributions from each delta sheet for a given zs."""
        # Initialize a full-sky kappa map with zeros.
        # The nside for this map should match the nside of the input delta_sheets.
        # If delta_sheets have varying nsides, this needs a more complex handling (resampling).
        # Assuming all delta_sheets have a consistent nside, which should be self.nside if they were made so.
        kappa_map_sum = np.zeros(self.npix, dtype=np.dtype(self.dtype_str))
        num_processed_sheets = 0

        args_list = [
            (data_path, wlen_int, self.dtype_str) # Pass dtype_str
            for data_path, wlen_int in zip(self.sheet_files, wlen_integrals)
            if wlen_int != 0 # Optimization: don't process sheets with zero weight (e.g. behind source)
        ]
        
        if not args_list:
            logging.warning("No sheets to process after filtering by wlen_integrals for current zs.")
            return kappa_map_sum # Return zero map if all weights are zero

        with Pool(processes=self.num_workers) as pool:
            results = pool.imap_unordered(process_delta_sheet, args_list)
            for delta_contrib in results:
                if delta_contrib is not None:
                    if delta_contrib.shape == kappa_map_sum.shape:
                        kappa_map_sum += delta_contrib
                        num_processed_sheets += 1
                    else:
                        logging.warning(f"Shape mismatch: kappa_map_sum {kappa_map_sum.shape}, delta_contrib {delta_contrib.shape}. Skipping contribution.")
        
        if num_processed_sheets == 0 and len(args_list) > 0:
            logging.error("All sheets failed to process or had shape mismatches. Kappa map will be zero.")
            return None # Indicate failure
            
        logging.info(f"Summed contributions from {num_processed_sheets} delta sheets.")
        return kappa_map_sum 