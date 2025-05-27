# ====================
# lensing_ssc/core/preprocessing/kappa.py
# ====================
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const, units as u
from astropy.cosmology import z_at_value
from multiprocessing import Pool
import re

from .utils import PerformanceMonitor


def compute_weight_function(chi: float, cosmo: FlatLambdaCDM, zs: float) -> float:
    """Compute the weight function for weak lensing convergence."""
    chis = cosmo.comoving_distance(zs).value  # in Mpc
    H0_for_formula = cosmo.H0.value / const.c.to(u.km/u.s).value
    z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    dchi_factor = np.clip(1.0 - (chi / chis), 0.0, None)
    return 1.5 * cosmo.Om0 * (H0_for_formula ** 2) * (1.0 + z) * chi * dchi_factor


def compute_wlen_integral(chi1: float, chi2: float, cosmo: FlatLambdaCDM, zs: float) -> float:
    """Compute the weak lensing integral for a single mass sheet."""
    if chi1 == chi2:
        return 0.0
    if chi1 < chi2:
        chi1, chi2 = chi2, chi1
        
    chi_mid = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
    delta_chi = chi1 - chi2
    return compute_weight_function(chi_mid, cosmo, zs) * delta_chi


def index_to_chi_pair(index: int, cosmo: FlatLambdaCDM) -> Tuple[float, float]:
    """Convert sheet index to comoving distance bounds using simple model."""
    # Simple model: a_i = 0.01 * i for sheets 20-100
    a_outer = 0.01 * index
    a_inner = 0.01 * (index + 1)
    
    if a_outer <= 0.0:
        a_outer = 0.001  # Avoid division by zero
    
    z_outer = (1.0 / a_outer) - 1.0
    z_inner = (1.0 / a_inner) - 1.0

    chi_further = cosmo.comoving_distance(z_outer).value
    chi_closer = cosmo.comoving_distance(z_inner).value
    
    return chi_closer, chi_further


def process_delta_sheet(args: Tuple[Path, float, str]) -> Optional[np.ndarray]:
    """Process a single delta sheet FITS file."""
    data_path, wlen_int, dtype_str = args
    logging.debug(f"Processing {data_path.name} with weight={wlen_int:.4e}")
    
    try:
        delta_map = hp.read_map(str(data_path), nest=None)
        delta_contribution = delta_map.astype(np.dtype(dtype_str)) * wlen_int
        return delta_contribution
    except Exception as e:
        logging.error(f"Failed to process {data_path.name}: {e}")
        return None


class KappaConstructor:
    """Construct kappa maps from delta sheets (no usmesh dependency)."""

    def __init__(self, mass_sheet_dir: Path, output_dir: Path, 
                 nside: int = 8192, zs_list: Optional[List[float]] = None, 
                 overwrite: bool = False, num_workers: Optional[int] = None, 
                 cosmo_params: Optional[Dict[str, float]] = None):
        """Initialize KappaConstructor."""
        self.mass_sheet_dir = Path(mass_sheet_dir)
        self.output_dir = Path(output_dir)
        self.monitor = PerformanceMonitor()
        
        # Extract seed from directory name
        self.seed = self._extract_seed_from_path(mass_sheet_dir)
        
        self.zs_list = zs_list or [0.5, 1.0, 1.5, 2.0, 2.5]
        self.overwrite = overwrite
        self.num_workers = num_workers
        
        # Initialize cosmology
        default_cosmo = {"H0": 67.74, "Om0": 0.309}
        if cosmo_params:
            default_cosmo.update(cosmo_params)
        self.cosmo = FlatLambdaCDM(H0=default_cosmo["H0"], Om0=default_cosmo["Om0"])
        
        self.dtype_str = "float32"
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.mass_sheet_dir.exists():
            raise FileNotFoundError(f"Mass sheet directory not found: {self.mass_sheet_dir}")

        # Find sheet files and extract indices
        self.sheet_files = sorted(self.mass_sheet_dir.glob("delta-sheet-*.fits"))
        if not self.sheet_files:
            raise ValueError(f"No delta-sheet-*.fits files found in {self.mass_sheet_dir}")
        
        self.sheet_indices = [int(f.stem.split("-")[-1]) for f in self.sheet_files]
        
        # Precompute chi_pairs using simple model
        self.chi_pairs = []
        for i in self.sheet_indices:
            try:
                self.chi_pairs.append(index_to_chi_pair(i, self.cosmo))
            except Exception as e:
                logging.warning(f"Error computing chi_pair for sheet {i}: {e}")
                self.chi_pairs.append((0.0, 0.0))  # Zero weight

        logging.info(f"Initialized KappaConstructor with {len(self.sheet_files)} sheets")

    def _extract_seed_from_path(self, path: Path) -> str:
        """Extract seed from path string."""
        path_str = str(path)
        match = re.search(r's(\d+)', path_str)
        return match.group(1) if match else "unknown"

    def compute_all_kappas(self) -> Dict[str, Any]:
        """Compute and save kappa maps for all source redshifts."""
        results = {}
        
        for zs in self.zs_list:
            with self.monitor.timer(f"kappa_zs_{zs}"):
                logging.info(f"Computing kappa for zs={zs}")
                result = self._compute_kappa_for_zs(zs)
                results[f"zs_{zs}"] = result
        
        summary = {
            "total_source_redshifts": len(self.zs_list),
            "successful": sum(1 for r in results.values() if r.get("success", False)),
            "failed": sum(1 for r in results.values() if not r.get("success", False)),
            "performance": self.monitor.get_summary()
        }
        
        logging.info(f"Kappa generation summary: {summary['successful']} successful, {summary['failed']} failed")
        return summary

    def _compute_kappa_for_zs(self, zs: float) -> Dict[str, Any]:
        """Compute kappa map for a single source redshift."""
        kappa_file_name = f"kappa_zs{zs}_s{self.seed}_nside{self.nside}.fits"
        kappa_file = self.output_dir / kappa_file_name
        
        if kappa_file.exists() and not self.overwrite:
            logging.info(f"Kappa map {kappa_file.name} already exists. Skipping.")
            return {"success": True, "skipped": True, "file": kappa_file}

        try:
            wlen_integrals = self._precompute_wlen_integrals(zs)
            kappa_map = self._compute_kappa_map(wlen_integrals)
            
            if kappa_map is None:
                return {"success": False, "error": "Failed to compute kappa map"}
            
            hp.write_map(str(kappa_file), kappa_map, dtype=np.dtype(self.dtype_str), 
                        nest=None, overwrite=self.overwrite)
            
            logging.info(f"Kappa map saved to {kappa_file.name}")
            return {"success": True, "file": kappa_file}
            
        except Exception as e:
            logging.error(f"Failed to compute kappa for zs={zs}: {e}")
            return {"success": False, "error": str(e)}

    def _precompute_wlen_integrals(self, zs: float) -> List[float]:
        """Precompute weak lensing integrals for all mass sheets."""
        integrals = []
        for chi_closer, chi_further in self.chi_pairs:
            if chi_closer == 0.0 and chi_further == 0.0:
                integrals.append(0.0)
            else:
                integral = compute_wlen_integral(chi_closer, chi_further, self.cosmo, zs)
                integrals.append(integral)
        return integrals

    def _compute_kappa_map(self, wlen_integrals: List[float]) -> Optional[np.ndarray]:
        """Compute the global kappa map."""
        kappa_map_sum = np.zeros(self.npix, dtype=np.dtype(self.dtype_str))
        
        # Filter out zero weights
        args_list = [
            (data_path, wlen_int, self.dtype_str)
            for data_path, wlen_int in zip(self.sheet_files, wlen_integrals)
            if abs(wlen_int) > 1e-15
        ]
        
        if not args_list:
            logging.warning("No sheets with non-zero weights")
            return kappa_map_sum

        num_processed = 0
        
        if self.num_workers and self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                results = pool.imap_unordered(process_delta_sheet, args_list)
                for delta_contrib in results:
                    if delta_contrib is not None and delta_contrib.shape == kappa_map_sum.shape:
                        kappa_map_sum += delta_contrib
                        num_processed += 1
        else:
            for args in args_list:
                delta_contrib = process_delta_sheet(args)
                if delta_contrib is not None and delta_contrib.shape == kappa_map_sum.shape:
                    kappa_map_sum += delta_contrib
                    num_processed += 1
        
        if num_processed == 0:
            logging.error("No sheets were successfully processed")
            return None
            
        logging.info(f"Processed {num_processed}/{len(args_list)} delta sheets")
        return kappa_map_sum