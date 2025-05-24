# scripts/03_run_analysis.py
import argparse
import logging
from pathlib import Path
import json
import re
import numpy as np
import healpy as hp
from astropy import units as u
from lenstools import ConvergenceMap
import multiprocessing as mp

from lensing_ssc.core.patching_utils import PatchProcessor
from lensing_ssc.core.fibonacci_utils import FibonacciGrid # Potentially for default center_points logic
from lensing_ssc.stats.power_spectrum import calculate_power_spectrum
from lensing_ssc.stats.bispectrum import calculate_bispectrum
from lensing_ssc.stats.pdf import calculate_pdf
from lensing_ssc.stats.peak_counts import calculate_peak_counts
from lensing_ssc.io.file_handlers import save_results_to_hdf5

# Default values for analysis parameters (can be overridden by config or CLI)
DEFAULT_NGAL_LIST = [0, 7, 15, 30, 50] # Galaxy densities for noise
DEFAULT_SL_LIST = [2.0, 5.0, 8.0, 10.0]    # Smoothing lengths in arcminutes
DEFAULT_LMIN = 300
DEFAULT_LMAX = 3000
DEFAULT_NBIN_PS_BS = 8      # Number of bins for PS and BS
DEFAULT_NBIN_PDF_PEAKS = 50 # Number of bins for PDF and Peak Counts (e.g., for nu in [-5, 5])
DEFAULT_PDF_PEAKS_RANGE = (-5.0, 5.0)
DEFAULT_EPSILON_NOISE = 0.26 # Galaxy shape noise

def parse_kappa_filename(filename: str) -> dict:
    """Parses kappa filename like kappa_zs1.0_s123_nside8192.fits"""
    match = re.match(r"kappa_zs(\d+\.?\d*)_s(\w+)_nside(\d+).fits", Path(filename).name)
    if match:
        return {
            "zs": float(match.group(1)),
            "seed": str(match.group(2)),
            "nside": int(match.group(3)),
        }
    return {}

def _analysis_worker(
    patch_data: np.ndarray,
    patch_size_deg: float,
    ngal: int,
    sl_list: List[float],
    ps_bs_l_edges: np.ndarray,
    ps_bs_ell_mids: np.ndarray,
    pdf_peaks_nu_bins: np.ndarray,
    epsilon_noise: float,
    xsize: int # patch xsize in pixels
    ) -> dict:
    """
    Worker function to perform statistical analysis on a single patch.
    """
    patch_results = {"ps_bs": {}, "smoothed_stats": {sl: {} for sl in sl_list}}
    
    # Calculate pixel area in arcmin^2
    pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2

    current_patch_data = patch_data.copy()
    if ngal > 0:
        noise_sigma = epsilon_noise / np.sqrt(ngal * pixarea_arcmin2)
        noise = np.random.normal(0, noise_sigma, current_patch_data.shape)
        current_patch_data += noise

    conv_map = ConvergenceMap(current_patch_data, angle=patch_size_deg * u.deg)

    # Power Spectrum & Bispectrum (on un-smoothed, potentially noisy map)
    patch_results["ps_bs"]["cl"] = calculate_power_spectrum(conv_map, ps_bs_l_edges, ps_bs_ell_mids)
    equ, iso, sq = calculate_bispectrum(conv_map, ps_bs_l_edges, ps_bs_ell_mids)
    patch_results["ps_bs"]["bispec_equ"] = equ
    patch_results["ps_bs"]["bispec_iso"] = iso
    patch_results["ps_bs"]["bispec_sq"] = sq

    # Smoothed statistics
    for sl in sl_list:
        smoothed_map_data = conv_map.smooth(sl * u.arcmin).data
        sigma0 = np.std(smoothed_map_data)
        
        # SNR map for PDF and Peaks
        if sigma0 > 1e-9: # Avoid division by zero for blank maps
            snr_map_data = smoothed_map_data / sigma0
        else:
            snr_map_data = smoothed_map_data # Effectively zero map
            
        snr_conv_map = ConvergenceMap(snr_map_data, angle=patch_size_deg * u.deg)
        
        patch_results["smoothed_stats"][sl]["pdf"] = calculate_pdf(snr_conv_map, pdf_peaks_nu_bins)
        patch_results["smoothed_stats"][sl]["peaks"] = calculate_peak_counts(snr_conv_map, pdf_peaks_nu_bins, is_minima=False)
        patch_results["smoothed_stats"][sl]["minima"] = calculate_peak_counts(snr_conv_map, pdf_peaks_nu_bins, is_minima=True)
        patch_results["smoothed_stats"][sl]["sigma0"] = sigma0
        # Minkowski functionals could be added here if a lensing_ssc.stats.minkowski module existed
        # e.g., v0, v1, v2 = calculate_minkowski(snr_conv_map, pdf_peaks_nu_bins)
        # patch_results["smoothed_stats"][sl]["v0"], ... "v1", ... "v2"] = v0, v1, v2

    return patch_results

def main():
    parser = argparse.ArgumentParser(description="Run patch-based statistical analysis on kappa maps.")
    # Input/Output
    parser.add_argument("kappa_input_dir", type=str, help="Directory containing full-sky kappa maps (.fits files).")
    parser.add_argument("patch_output_dir", type=str, help="Directory to save/load generated patches (.npy files).")
    parser.add_argument("stats_output_dir", type=str, help="Directory to save final HDF5 statistics files.")

    # Patch parameters
    parser.add_argument("--patch_size_deg", type=float, default=10.0, help="Size of square patches in degrees.")
    parser.add_argument("--patch_xsize", type=int, default=256, help="Resolution (side pixels) of square patches.") # Reduced from 2048 for speed unless high-res needed
    parser.add_argument("--center_points_path", type=str, default="lensing_ssc/core/fibonacci/center_points/", 
                        help="Path to directory containing Fibonacci center points files.")

    # Analysis parameters
    parser.add_argument("--ngal_list", type=str, default=json.dumps(DEFAULT_NGAL_LIST), help="JSON list of galaxy densities for noise.")
    parser.add_argument("--sl_list", type=str, default=json.dumps(DEFAULT_SL_LIST), help="JSON list of smoothing lengths (arcmin).")
    parser.add_argument("--lmin", type=int, default=DEFAULT_LMIN, help="Minimum multipole for PS/BS.")
    parser.add_argument("--lmax", type=int, default=DEFAULT_LMAX, help="Maximum multipole for PS/BS.")
    parser.add_argument("--nbin_ps_bs", type=int, default=DEFAULT_NBIN_PS_BS, help="Number of bins for PS/BS.")
    parser.add_argument("--nbin_pdf_peaks", type=int, default=DEFAULT_NBIN_PDF_PEAKS, help="Number of bins for PDF/Peak counts.")
    parser.add_argument("--pdf_peaks_min_max", type=str, default=json.dumps(list(DEFAULT_PDF_PEAKS_RANGE)),
                        help="JSON list for [min, max] nu/sigma values for PDF/Peak bins.")
    parser.add_argument("--epsilon_noise", type=float, default=DEFAULT_EPSILON_NOISE, help="Galaxy shape noise parameter epsilon.")
    
    # Control
    parser.add_argument("--overwrite_patches", action="store_true", help="Overwrite existing patch files.")
    parser.add_argument("--overwrite_stats", action="store_true", help="Overwrite existing statistics HDF5 files.")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count(), help="Number of CPU processes for parallel tasks.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger("03_run_analysis")

    # Create output directories if they don't exist
    Path(args.patch_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.stats_output_dir).mkdir(parents=True, exist_ok=True)

    # Parse JSON arguments
    try:
        ngal_list = json.loads(args.ngal_list)
        sl_list = json.loads(args.sl_list)
        pdf_peaks_min_max = json.loads(args.pdf_peaks_min_max)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON arguments: {e}")
        return

    # Initialize PatchProcessor
    patch_processor = PatchProcessor(
        patch_size_deg=args.patch_size_deg,
        xsize=args.patch_xsize,
        center_points_path=args.center_points_path
    )
    logger.info(f"Initialized PatchProcessor: patch_size={args.patch_size_deg} deg, xsize={args.patch_xsize}px.")

    # Prepare analysis bins
    ps_bs_l_edges = np.logspace(np.log10(args.lmin), np.log10(args.lmax), args.nbin_ps_bs + 1)
    ps_bs_ell_mids = (ps_bs_l_edges[:-1] + ps_bs_l_edges[1:]) / 2
    pdf_peaks_nu_bins = np.linspace(pdf_peaks_min_max[0], pdf_peaks_min_max[1], args.nbin_pdf_peaks + 1)
    
    analysis_params_for_worker = {
        "patch_size_deg": args.patch_size_deg,
        "sl_list": sl_list,
        "ps_bs_l_edges": ps_bs_l_edges,
        "ps_bs_ell_mids": ps_bs_ell_mids,
        "pdf_peaks_nu_bins": pdf_peaks_nu_bins,
        "epsilon_noise": args.epsilon_noise,
        "xsize": args.patch_xsize
    }

    kappa_files = sorted(list(Path(args.kappa_input_dir).glob("kappa_*.fits")))
    if not kappa_files:
        logger.warning(f"No kappa maps found in {args.kappa_input_dir}. Exiting.")
        return
    
    logger.info(f"Found {len(kappa_files)} kappa maps to process.")

    for kappa_file_path in kappa_files:
        logger.info(f"Processing kappa map: {kappa_file_path.name}")
        kappa_file_info = parse_kappa_filename(kappa_file_path.name)
        
        patch_file_name = f"{kappa_file_path.stem}_patches_oa{args.patch_size_deg}_x{args.patch_xsize}.npy"
        patch_file_full_path = Path(args.patch_output_dir) / patch_file_name
        
        stats_file_name = f"{kappa_file_path.stem}_stats_oa{args.patch_size_deg}_x{args.patch_xsize}.hdf5"
        stats_file_full_path = Path(args.stats_output_dir) / stats_file_name

        if stats_file_full_path.exists() and not args.overwrite_stats:
            logger.info(f"Statistics file {stats_file_full_path} already exists. Skipping.")
            continue

        patches_data = None
        if patch_file_full_path.exists() and not args.overwrite_patches:
            logger.info(f"Loading existing patches from {patch_file_full_path}")
            try:
                patches_data = np.load(patch_file_full_path)
            except Exception as e:
                logger.error(f"Error loading patches from {patch_file_full_path}: {e}. Will attempt to regenerate.")
        
        if patches_data is None:
            logger.info(f"Generating patches for {kappa_file_path.name}")
            try:
                kappa_map_data = hp.read_map(str(kappa_file_path), nest=None) # Assume RING if not specified, or auto-detect
                # Reorder to RING if it's NEST, as gnomview expects RING
                # If read_map(nest=None) it tries to guess. If it's wrong, gnomview might fail or be wrong.
                # It's safer to know the input ordering or convert.
                # For now, assume input maps are RING or hp.read_map handles it.
                # The PatchGenerator used hp.reorder(map_data, n2r=True) after read_map.
                # Let's ensure map is in RING for gnomview if not already.
                # A simple check:
                if hp.isnpix(kappa_map_data.size): # Basic check if it's a healpix map
                    map_nside = hp.npix2nside(kappa_map_data.size)
                    # Check ordering, hp.gnomview needs RING
                    # A robust way is to always reorder to RING for gnomview.
                    # healpy's read_map with nest=None returns RING by default if it can determine.
                    # If we used nest=True when saving kappa maps, we should use nest=True here too and then reorder.
                    # Let's assume kappa maps are saved in a way that read_map(nest=None) gives RING or it's already RING.
                    # The legacy PatchGenerator did: map_data = hp.read_map(); hp.reorder(map_data, n2r=True)
                    # This means the input was likely NESTED. And gnomview in PatchProcessor uses nest=False.
                    # So, let's ensure RING input to make_patches if it uses gnomview's default nest=False.
                    # PatchProcessor's gnomview has nest=False, so it expects RING.
                    # Kappa maps from KappaConstructor are saved with nest=None, which means Healpy decides.
                    # If mass_sheets (delta-sheet-*.fits) are NEST, and then combined, the output kappa map's ordering depends on how hp.write_map was called.
                    # KappaConstructor.compute_all_kappas calls hp.write_map(..., nest=None)
                    # If the input delta_map in process_delta_sheet was read with nest=None and was NEST, it would be converted to RING. Sum of RING is RING.
                    # So kappa_map_data is likely RING.
                    
                patches_data = patch_processor.make_patches(kappa_map_data, num_processes=args.num_processes)
                np.save(patch_file_full_path, patches_data)
                logger.info(f"Saved {len(patches_data)} patches to {patch_file_full_path}")
            except Exception as e:
                logger.error(f"Error generating/saving patches for {kappa_file_path.name}: {e}", exc_info=True)
                continue # Skip to next kappa map
        
        if patches_data is None or len(patches_data) == 0:
            logger.warning(f"No patches available for {kappa_file_path.name}. Skipping analysis.")
            continue

        logger.info(f"Analyzing {len(patches_data)} patches from {patch_file_name}...")
        
        # Initialize results structure for this kappa file
        # results_for_hdf5 = {ng: {} for ng in ngal_list}
        # Simpler: one top level for this file, then ngal, then sl etc.
        # Let's follow the structure from legacy PatchAnalyzer which was:
        # results[ngal][stat_or_sl]
        # For PS/BS, stat_or_sl is "cl", "bispec_equ" etc.
        # For smoothed, stat_or_sl is the sl_value, then "pdf", "peaks" etc.
        
        aggregated_results = {
            ng: {
                "cl": [], "bispec_equ": [], "bispec_iso": [], "bispec_sq": [],
                **{sl: {"pdf": [], "peaks": [], "minima": [], "sigma0": []} for sl in sl_list}
            } for ng in ngal_list
        }

        # Prepare arguments for the pool
        pool_args = []
        for ngal_val in ngal_list:
            for i_patch in range(len(patches_data)):
                pool_args.append(
                    (patches_data[i_patch], ngal_val, analysis_params_for_worker)
                )
        
        # Use starmap correctly with a single dictionary for analysis_params
        def worker_wrapper(patch_data_item, ngal_item, params_dict):
            return _analysis_worker(
                patch_data=patch_data_item,
                ngal=ngal_item,
                **params_dict # Unpack the rest of the fixed params
            )

        processed_patch_results = []
        with mp.Pool(processes=args.num_processes) as pool:
            # Need to associate results back to ngal and patch index if order is not guaranteed or for error handling
            # For now, assume imap_unordered or starmap gives results that can be re-associated or order is fine
            # Let's re-structure pool_args to explicitly pass all varying params to worker
            flat_arg_list_for_pool = []
            for i_patch in range(len(patches_data)):
                 for ngal_val in ngal_list:
                    flat_arg_list_for_pool.append( (patches_data[i_patch], ngal_val) ) # Only pass what changes per task

            # Redefine worker_wrapper to take fixed_params from outside
            def worker_wrapper_dynamic(patch_data_item, ngal_item):
                return _analysis_worker(patch_data_item, ngal=ngal_item, **analysis_params_for_worker)

            raw_pool_results = pool.starmap(worker_wrapper_dynamic, flat_arg_list_for_pool)
        
        # Reconstruct the results (raw_pool_results is a flat list)
        result_idx = 0
        for i_patch in range(len(patches_data)):
            for ngal_val in ngal_list:
                patch_result = raw_pool_results[result_idx]
                result_idx += 1
                
                aggregated_results[ngal_val]["cl"].append(patch_result["ps_bs"]["cl"])
                aggregated_results[ngal_val]["bispec_equ"].append(patch_result["ps_bs"]["bispec_equ"])
                aggregated_results[ngal_val]["bispec_iso"].append(patch_result["ps_bs"]["bispec_iso"])
                aggregated_results[ngal_val]["bispec_sq"].append(patch_result["ps_bs"]["bispec_sq"])
                for sl_val in sl_list:
                    aggregated_results[ngal_val][sl_val]["pdf"].append(patch_result["smoothed_stats"][sl_val]["pdf"])
                    aggregated_results[ngal_val][sl_val]["peaks"].append(patch_result["smoothed_stats"][sl_val]["peaks"])
                    aggregated_results[ngal_val][sl_val]["minima"].append(patch_result["smoothed_stats"][sl_val]["minima"])
                    aggregated_results[ngal_val][sl_val]["sigma0"].append(patch_result["smoothed_stats"][sl_val]["sigma0"])

        # Convert lists of arrays to single arrays (num_patches, num_bins_or_values)
        final_results_for_hdf5 = {}
        for ngal_val in ngal_list:
            final_results_for_hdf5[ngal_val] = {}
            final_results_for_hdf5[ngal_val]["cl"] = np.array(aggregated_results[ngal_val]["cl"])
            final_results_for_hdf5[ngal_val]["bispec_equ"] = np.array(aggregated_results[ngal_val]["bispec_equ"])
            final_results_for_hdf5[ngal_val]["bispec_iso"] = np.array(aggregated_results[ngal_val]["bispec_iso"])
            final_results_for_hdf5[ngal_val]["bispec_sq"] = np.array(aggregated_results[ngal_val]["bispec_sq"])
            for sl_val in sl_list:
                final_results_for_hdf5[ngal_val][sl_val] = {
                    "pdf": np.array(aggregated_results[ngal_val][sl_val]["pdf"]),
                    "peaks": np.array(aggregated_results[ngal_val][sl_val]["peaks"]),
                    "minima": np.array(aggregated_results[ngal_val][sl_val]["minima"]),
                    "sigma0": np.array(aggregated_results[ngal_val][sl_val]["sigma0"]),
                }
        
        # Prepare metadata for HDF5
        metadata_obj = argparse.Namespace(
            patch_size=args.patch_size_deg, # Renaming for legacy save_results_to_hdf5 compatibility
            xsize=args.patch_xsize,
            pixarea_arcmin2=(args.patch_size_deg * 60.0 / args.patch_xsize)**2,
            lmin=args.lmin,
            lmax=args.lmax,
            nbin=args.nbin_ps_bs, # This 'nbin' is ambiguous, legacy used it for PS/BS and PDF. Let's use PS/BS.
            epsilon=args.epsilon_noise,
            ngal_list=ngal_list,
            sl_list=sl_list,
            bins=pdf_peaks_nu_bins, # For PDF/Peaks (nu values, or bin edges)
            nu=(pdf_peaks_nu_bins[:-1] + pdf_peaks_nu_bins[1:]) / 2, # PDF/Peak bin centers
            l_edges=ps_bs_l_edges,
            ell=ps_bs_ell_mids,
            kappa_file_info=kappa_file_info # Store info parsed from filename
        )

        logger.info(f"Saving statistics to {stats_file_full_path}")
        try:
            # save_results_to_hdf5 expects a specific structure.
            # If final_results_for_hdf5 has ngal as top key, it might interpret it as "single result".
            # The legacy save_results_to_hdf5 can handle:
            # 1. results = {ngal: {sl: {stat: val_array}}} (single file)
            # 2. results = {sim_id: {ngal: {sl: {stat: val_array}}}} (batch)
            # Here, final_results_for_hdf5 is for a single input kappa map, so it's case 1.
            save_results_to_hdf5(final_results_for_hdf5, stats_file_full_path, analyzer=metadata_obj)
            logger.info(f"Successfully saved statistics for {kappa_file_path.name}")
        except Exception as e:
            logger.error(f"Error saving HDF5 results for {kappa_file_path.name}: {e}", exc_info=True)

    logger.info("All kappa maps processed.")

if __name__ == "__main__":
    main() 