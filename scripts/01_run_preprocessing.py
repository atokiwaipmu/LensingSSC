# scripts/01_run_preprocessing.py
import argparse
import logging
from pathlib import Path

from lensing_ssc.core.preprocessing import MassSheetProcessor, ProcessingConfig


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
    parser.add_argument(
        "--sheet-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Process only sheets in range [START, END)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation step"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )

    data_directory = Path(args.datadir)

    if not data_directory.exists():
        logging.error(f"Data directory '{data_directory}' does not exist. Please provide a valid path.")
        return

    usmesh_dir = data_directory / "usmesh"
    if not usmesh_dir.exists():
        logging.error(f"Required subdirectory 'usmesh' not found in '{data_directory}'")
        return

    logging.info(f"Starting mass sheet preprocessing for data in: {data_directory}")
    logging.info(f"Overwrite existing files: {args.overwrite}")
    if args.dry_run:
        logging.info("DRY-RUN MODE: No actual processing will occur")

    try:
        # Create configuration
        config = ProcessingConfig(overwrite=args.overwrite)
        if args.sheet_range:
            config.sheet_range = tuple(args.sheet_range)
            logging.info(f"Sheet range: {config.sheet_range}")
            
        processor = MassSheetProcessor(datadir=data_directory, config=config)
        
        # Validate data structure (unless skipped)
        if not args.skip_validation:
            logging.info("Validating data structure...")
            try:
                if not processor.validate_data():
                    logging.error("Data validation failed")
                    return
                logging.info("Data validation passed")
            except Exception as e:
                logging.error(f"Data validation error: {e}")
                logging.warning("You can skip validation with --skip-validation if the data is known to be valid")
                return
        
        # Show processing information
        info = processor.get_processing_info()
        
        print("\n" + "="*60)
        print("PREPROCESSING PLAN")
        print("="*60)
        print(f"Data directory: {processor.datadir}")
        print(f"Seed: {info.get('seed', 'unknown')}")
        print(f"Total sheets available: {info.get('total_sheets', 0)}")
        print(f"Sheets to process: {info.get('sheets_to_process', 0)}")
        print(f"Sheet range: {info.get('sheet_range', 'N/A')}")
        print(f"Output directory: {processor.output_dir}")
        print(f"Overwrite existing: {processor.config.overwrite}")
        
        if info.get('existing_files'):
            print(f"Existing output files: {len(info['existing_files'])}")
        else:
            print("Existing output files: None")
            
        # Show estimated processing info
        total_sheets = info.get('sheets_to_process', 0)
        if total_sheets > 0:
            # Rough estimate: 1-5 minutes per sheet for large datasets
            estimated_minutes = total_sheets * 3  # Conservative estimate
            hours = estimated_minutes // 60
            minutes = estimated_minutes % 60
            print(f"Estimated processing time: {hours}h {minutes}m (rough estimate)")
        
        print("="*60)
        
        if args.dry_run:
            print("\nDRY-RUN COMPLETE: No actual processing performed.")
            logging.info("Dry-run completed successfully.")
            return
        
        # Run actual preprocessing
        logging.info("Starting actual preprocessing...")
        result = processor.preprocess(resume=args.resume)
        
        # Print results
        print("\n" + "="*60)
        print("PREPROCESSING RESULTS")
        print("="*60)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Total sheets: {result.get('total_sheets', 0)}")
        print(f"Previously completed: {result.get('previously_completed', 0)}")
        print(f"Newly processed: {result.get('newly_processed', 0)}")
        print(f"Failed: {result.get('failed', 0)}")
        
        if result.get('failed_sheets'):
            print(f"Failed sheets: {result['failed_sheets']}")
            
        total_time = result.get('total_time', 0)
        if total_time > 0:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"Total processing time: {hours}h {minutes}m {seconds}s")
            
        avg_time = result.get('average_time_per_sheet', 0)
        if avg_time > 0:
            print(f"Average time per sheet: {avg_time:.2f}s")
        
        print("="*60)
        
        if result.get("status") == "complete":
            logging.info("Mass sheet preprocessing completed successfully.")
        else:
            logging.warning(f"Preprocessing completed with status: {result.get('status')}")
            
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
        logging.error("Preprocessing failed.")


if __name__ == "__main__":
    main()