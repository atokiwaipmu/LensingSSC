# scripts/02_run_kappa_generation.py
import argparse
import logging
from pathlib import Path
import json # For parsing list of floats

from lensing_ssc.core.preprocessing_utils import KappaConstructor, MassSheetProcessor # MassSheetProcessor to get usmesh_attrs

def main():
    """Main function to run the kappa map generation pipeline."""
    parser = argparse.ArgumentParser(description="Run kappa map generation from mass sheets.")
    parser.add_argument(
        "basedir",
        type=str,
        help="Base directory containing the 'mass_sheets' subdirectory and where 'kappa_maps' will be created."
    )
    parser.add_argument(
        "--zs_list",
        type=str,
        default="[0.5, 1.0, 1.5, 2.0, 2.5]",
        help='JSON string of a list of source redshifts (e.g., "[0.5, 1.0, 2.0]").'
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=8192,
        help="NSIDE parameter for the output Healpix kappa maps (default: 8192)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing kappa map files. Defaults to False."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for processing delta sheets (default: all available CPUs)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=args.log_level.upper(), 
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    base_directory = Path(args.basedir)
    mass_sheet_directory = base_directory / "mass_sheets"
    kappa_output_directory = base_directory / "kappa_maps" # Changed from "kappa" to "kappa_maps" for clarity

    if not mass_sheet_directory.exists():
        logging.error(f"Mass sheet directory '{mass_sheet_directory}' not found. Please run 01_run_preprocessing.py first or provide a valid path.")
        return

    try:
        zs_list = json.loads(args.zs_list)
        if not isinstance(zs_list, list) or not all(isinstance(zs, (float, int)) for zs in zs_list):
            raise ValueError("zs_list must be a list of numbers.")
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Invalid format for --zs_list. Expected a JSON list of numbers (e.g., \"[0.5, 1.0]\"). Error: {e}")
        return

    logging.info(f"Starting kappa map generation from mass sheets in: {mass_sheet_directory}")
    logging.info(f"Output directory for kappa maps: {kappa_output_directory}")
    logging.info(f"Source redshifts (zs_list): {zs_list}")
    logging.info(f"NSIDE for kappa maps: {args.nside}")
    logging.info(f"Overwrite existing files: {args.overwrite}")
    logging.info(f"Number of workers: {args.num_workers if args.num_workers is not None else 'All CPUs'}")

    try:
        # Need usmesh_attrs for KappaConstructor. We can get this by instantiating MassSheetProcessor
        # on the parent directory of mass_sheet_directory (which should contain 'usmesh').
        # This assumes basedir contains the original 'usmesh' or is structured like 'run0/' etc.
        # If basedir *is* the directory containing 'usmesh', then MassSheetProcessor(basedir) is correct.
        # Let's assume basedir is the one containing 'usmesh' directly.
        try:
            logging.debug(f"Attempting to load usmesh attributes from: {base_directory}")
            # We don't need to run preprocess, just initialize to load attributes.
            # Temporarily set log level high for this internal step if desired.
            temp_msp = MassSheetProcessor(datadir=base_directory, overwrite=False) 
            usmesh_attrs = temp_msp.msheets.attrs
            logging.info("Successfully loaded usmesh attributes.")
        except Exception as e:
            logging.error(f"Failed to load usmesh_attrs using MassSheetProcessor from '{base_directory}'. These are required for KappaConstructor. Error: {e}")
            logging.error("Kappa map generation failed.")
            return

        kappa_constructor = KappaConstructor(
            mass_sheet_dir=mass_sheet_directory,
            output_dir=kappa_output_directory,
            usmesh_attrs=usmesh_attrs,
            nside=args.nside,
            zs_list=zs_list,
            overwrite=args.overwrite,
            num_workers=args.num_workers
        )
        kappa_constructor.compute_all_kappas()
        logging.info("Kappa map generation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during kappa map generation: {e}", exc_info=True)
        logging.error("Kappa map generation failed.")

if __name__ == "__main__":
    main() 