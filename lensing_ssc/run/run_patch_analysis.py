import argparse
import logging
import glob
from pathlib import Path

import numpy as np

from lensing_ssc.core.patch.statistics.analyzer import PatchAnalyzer
from lensing_ssc.utils.io import save_results_to_hdf5


def main():
    parser = argparse.ArgumentParser(description="Process convergence maps and compute statistics.")
    parser.add_argument("--box_type", type=str, default="tiled", help="Type of box to process (tiled or bigbox)")
    parser.add_argument("--zs", type=float, default=0.5, help="Source redshift to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Paths setup
    data_dir = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/data/patches")
    output_base_dir = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/output")
    
    # Initialize analyzer
    patch_analyzer = PatchAnalyzer()

    save_dir = output_base_dir / args.box_type / "stats"
    logging.info(f"Starting processing for box_type={args.box_type}, saving to {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find all patch files
    # Format zs value properly to match directory structure
    zs_str = f"{args.zs:.1f}".rstrip('0').rstrip('.') if args.zs == int(args.zs) else f"{args.zs:.1f}"
    search_path = str(data_dir / args.box_type / f"zs{zs_str}" / "noiseless" / "*.npy")
    patches_kappa_paths = glob.glob(search_path)
    logging.info(f"Searching in path: {search_path}")
    logging.info(f"Found {len(patches_kappa_paths)} files for zs={args.zs} in box_type={args.box_type}")
    
    if not patches_kappa_paths:
        logging.warning(f"No .npy files found for zs={args.zs} in {args.box_type}. Exiting.")
        return

    for f in patches_kappa_paths:
        # Construct the save path
        filename = Path(f).name
        save_filename = filename.replace("patches", "stats").replace("_noiseless", "").replace(".npy", ".h5")
        save_path = save_dir / save_filename

        if save_path.exists() and not args.overwrite:
            logging.info(f"Skipping {Path(f).name} as {save_path} already exists.")
            continue

        logging.info(f"Processing {Path(f).name}")
        patches_kappa = np.load(f, mmap_mode='r')
        results = patch_analyzer.process_patches(patches_kappa)

        logging.info(f"Saving results to {save_path}")
        # メタデータも一緒に保存するように修正
        save_results_to_hdf5(results, save_path, analyzer=patch_analyzer)


if __name__ == "__main__":
    main()