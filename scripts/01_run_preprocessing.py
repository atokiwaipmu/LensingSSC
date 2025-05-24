# scripts/01_run_preprocessing.py
import argparse
import logging
from pathlib import Path

from lensing_ssc.core.preprocessing_utils import MassSheetProcessor

def main():
    """Main function to run the mass sheet preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run mass sheet preprocessing.")
    parser.add_argument(
        "datadir",
        type=str,
        help="Directory containing the input data (e.g., usmesh files)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files. Defaults to False."
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

    data_directory = Path(args.datadir)

    if not data_directory.exists() or not (data_directory / "usmesh").exists():
        logging.error(f"Data directory '{data_directory}' or subdirectory 'usmesh' not found. Please provide a valid path.")
        return

    logging.info(f"Starting mass sheet preprocessing for data in: {data_directory}")
    logging.info(f"Overwrite existing files: {args.overwrite}")

    try:
        processor = MassSheetProcessor(datadir=data_directory, overwrite=args.overwrite)
        processor.preprocess() # This method handles the actual processing and saving
        logging.info("Mass sheet preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
        logging.error("Preprocessing failed.")

if __name__ == "__main__":
    main() 