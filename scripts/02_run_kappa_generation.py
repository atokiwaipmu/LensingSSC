# scripts/02_run_kappa_generation.py
import argparse
import logging
from pathlib import Path
import json

from lensing_ssc.core.preprocessing.kappa import KappaConstructor


def main():
    """Main function to run kappa map generation."""
    parser = argparse.ArgumentParser(description="Generate kappa maps from mass sheets.")
    parser.add_argument(
        "mass_sheet_dir",
        type=str,
        help="Directory containing delta-sheet-*.fits files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for kappa maps (default: mass_sheet_dir/../kappa_maps)"
    )
    parser.add_argument(
        "--zs-list",
        type=str,
        default="[0.5, 1.0, 1.5, 2.0, 2.5]",
        help='Source redshifts as JSON list'
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=8192,
        help="NSIDE for output maps"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without processing"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    mass_sheet_dir = Path(args.mass_sheet_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = mass_sheet_dir.parent / "kappa_maps"

    if not mass_sheet_dir.exists():
        logging.error(f"Mass sheet directory not found: {mass_sheet_dir}")
        return

    try:
        zs_list = json.loads(args.zs_list)
    except json.JSONDecodeError:
        logging.error("Invalid zs-list format. Use JSON like '[0.5, 1.0]'")
        return

    # Check available files
    sheet_files = list(mass_sheet_dir.glob("delta-sheet-*.fits"))
    if not sheet_files:
        logging.error(f"No delta-sheet-*.fits files found in {mass_sheet_dir}")
        return

    if args.dry_run:
        logging.info("DRY-RUN MODE")

    print(f"\nKappa Generation Plan:")
    print(f"Mass sheets: {mass_sheet_dir} ({len(sheet_files)} files)")
    print(f"Output: {output_dir}")
    print(f"Source redshifts: {zs_list}")
    print(f"NSIDE: {args.nside}")
    print(f"Workers: {args.num_workers or 'Auto'}")

    if args.dry_run:
        print("\nDRY-RUN COMPLETE")
        return

    try:
        constructor = KappaConstructor(
            mass_sheet_dir=mass_sheet_dir,
            output_dir=output_dir,
            nside=args.nside,
            zs_list=zs_list,
            overwrite=args.overwrite,
            num_workers=args.num_workers
        )
        
        result = constructor.compute_all_kappas()
        
        print(f"\nResults:")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        
        if result['failed'] == 0:
            logging.info("Kappa generation completed successfully")
        else:
            logging.warning(f"Completed with {result['failed']} failures")

    except Exception as e:
        logging.error(f"Kappa generation failed: {e}")


if __name__ == "__main__":
    main()