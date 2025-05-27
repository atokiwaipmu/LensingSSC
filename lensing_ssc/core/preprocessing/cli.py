# ====================
# lensing_ssc/core/preprocessing/cli.py
# ====================
import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

from .config import ProcessingConfig
from .utils import setup_logging, format_duration, PerformanceMonitor
from .processing import MassSheetProcessor
from .validation import DataValidator


class PreprocessingCLI:
    """Enhanced command-line interface for mass sheet preprocessing."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Run optimized mass sheet preprocessing pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic preprocessing
  python -m lensing_ssc.cli preprocess data/simulation_s100/
  
  # With custom configuration
  python -m lensing_ssc.cli preprocess data/simulation_s100/ --config config.yaml
  
  # Resume from checkpoint
  python -m lensing_ssc.cli preprocess data/simulation_s100/ --resume
  
  # Process specific sheet range
  python -m lensing_ssc.cli preprocess data/simulation_s100/ --sheet-range 20 50
  
  # Validation only
  python -m lensing_ssc.cli validate data/simulation_s100/
            """
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Preprocess command
        preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing')
        self._add_preprocess_args(preprocess_parser)
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate data structure')
        self._add_validate_args(validate_parser)
        
        return parser
        
    def _add_preprocess_args(self, parser: argparse.ArgumentParser):
        """Add preprocessing-specific arguments."""
        parser.add_argument(
            "datadir",
            type=Path,
            help="Directory containing input data (must have 'usmesh' subdirectory)"
        )
        
        parser.add_argument(
            "--config", "-c",
            type=Path,
            help="Configuration file (YAML/JSON)"
        )
        
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing processed files"
        )
        
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from checkpoint if available"
        )
        
        parser.add_argument(
            "--sheet-range",
            type=int,
            nargs=2,
            metavar=("START", "END"),
            help="Process only sheets in range [START, END)"
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Logging level (default: INFO)"
        )
        
        parser.add_argument(
            "--log-file",
            type=Path,
            help="Log to file in addition to console"
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without actually processing"
        )
        
        parser.add_argument(
            "--skip-validation",
            action="store_true",
            help="Skip data validation step"
        )
        
    def _add_validate_args(self, parser: argparse.ArgumentParser):
        """Add validation-specific arguments."""
        parser.add_argument(
            "datadir",
            type=Path,
            help="Directory containing data to validate"
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Logging level (default: INFO)"
        )
        
    def run_preprocess(self, args: argparse.Namespace) -> int:
        """Run the preprocessing pipeline."""
        try:
            # Setup logging
            setup_logging(args.log_level, args.log_file)
            
            # Load configuration
            config = self._load_config(args)
            
            # Validate data directory
            if not self._validate_data_directory(args.datadir):
                return 1
                
            # Initialize processor
            processor = MassSheetProcessor(
                datadir=args.datadir,
                config=config
            )
            
            # Validate data structure (unless skipped)
            if not args.skip_validation:
                with self.performance_monitor.timer("validation"):
                    if not processor.validate_data():
                        logging.error("Data validation failed")
                        return 1
                        
            # Show dry-run information
            if args.dry_run:
                return self._show_dry_run_info(processor)
                
            # Run preprocessing
            with self.performance_monitor.timer("preprocessing"):
                success = processor.preprocess(resume=args.resume)
                
            # Log performance summary
            self.performance_monitor.log_summary()
            
            return 0 if success.get("status") == "complete" else 1
            
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
            return 130
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}", exc_info=True)
            return 1
            
    def run_validate(self, args: argparse.Namespace) -> int:
        """Run data validation."""
        try:
            setup_logging(args.log_level)
            
            if not self._validate_data_directory(args.datadir):
                return 1
                
            validator = DataValidator()
            
            with self.performance_monitor.timer("validation"):
                success = validator.validate_data(args.datadir)
                
            self.performance_monitor.log_summary()
            
            return 0 if success else 1
            
        except Exception as e:
            logging.error(f"Validation failed: {e}", exc_info=True)
            return 1
            
    def _load_config(self, args: argparse.Namespace) -> ProcessingConfig:
        """Load configuration from file or command line arguments."""
        if args.config and args.config.exists():
            config = ProcessingConfig.from_file(args.config)
        else:
            config = ProcessingConfig()
            
        # Override with command line arguments
        if hasattr(args, 'sheet_range') and args.sheet_range:
            config.sheet_range = tuple(args.sheet_range)
        if hasattr(args, 'overwrite'):
            config.overwrite = args.overwrite
            
        return config
        
    def _validate_data_directory(self, datadir: Path) -> bool:
        """Validate that data directory exists and has required structure."""
        if not datadir.exists():
            logging.error(f"Data directory '{datadir}' does not exist")
            return False
            
        usmesh_dir = datadir / "usmesh"
        if not usmesh_dir.exists():
            logging.error(f"Required subdirectory 'usmesh' not found in '{datadir}'")
            return False
            
        return True
        
    def _show_dry_run_info(self, processor) -> int:
        """Show what would be processed in dry-run mode."""
        try:
            info = processor.get_processing_info()
            
            print("Dry Run - Processing Plan:")
            print(f"  Data directory: {processor.datadir}")
            print(f"  Seed: {info.get('seed', 'unknown')}")
            print(f"  Total sheets available: {info.get('total_sheets', 0)}")
            print(f"  Sheets to process: {info.get('sheets_to_process', 0)}")
            print(f"  Sheet range: {info.get('sheet_range', 'N/A')}")
            print(f"  Output directory: {processor.output_dir}")
            print(f"  Overwrite existing: {processor.config.overwrite}")
            
            if info.get('existing_files'):
                print(f"  Existing output files: {len(info['existing_files'])}")
                
            return 0
            
        except Exception as e:
            logging.error(f"Failed to generate dry-run info: {e}")
            return 1
            
    def main(self, argv: Optional[List[str]] = None) -> int:
        """Main entry point."""
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        if not args.command:
            parser.print_help()
            return 1
            
        if args.command == 'preprocess':
            return self.run_preprocess(args)
        elif args.command == 'validate':
            return self.run_validate(args)
        else:
            logging.error(f"Unknown command: {args.command}")
            return 1


def main():
    """Entry point for command line usage."""
    cli = PreprocessingCLI()
    sys.exit(cli.main())


if __name__ == "__main__":
    main()