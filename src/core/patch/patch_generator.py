

import logging
from pathlib import Path
from typing import List, Tuple
import os
from matplotlib.pyplot import box
import numpy as np
import healpy as hp
import multiprocessing as mp

from src.core.patch.patch_processor import PatchProcessor
from src.utils.info_extractor import InfoExtractor
from lenstools import ConvergenceMap
from astropy import units as u

class PatchGenerator:
    def __init__(self, pp: PatchProcessor, 
                 zs_list,
                 data_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/data", 
                 overwrite=False):
        self.data_dir = Path(data_dir)
        self.zs_list = zs_list
        self.pp = pp
        self.overwrite = overwrite

        self.kappa_paths = sorted(self.data_dir.glob("fullsky/*/zs*/*.fits")) # fullsky/seed/zs*/kappa_zs*_s*.fits
        self.output_dir = self.data_dir / f"patch_oa{int(self.pp.patch_size_deg)}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dir_initialize()

    def dir_initialize(self):
        for zs in self.zs_list:
            for box_type in ["bigbox", "tiled"]:
                output_dir = self.output_dir / box_type / f"zs{zs}" 
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory {output_dir}")
        

    def run(self):
        for kappa_path in self.kappa_paths:
            info = InfoExtractor.extract_info_from_path(kappa_path)
            box_type = kappa_path.parts[-3]
            input_map = self.load_map(kappa_path)
            output_dir = self.output_dir / box_type / f"zs{info['redshift']}" 
            logging.info(f"Output directory: {output_dir}")
            output_path = self.generate_fname(kappa_path, output_dir)
            if os.path.exists(output_path) and not self.overwrite:
                logging.info(f"Skipping {output_path}")
                continue
            logging.info(f"Generating patches for seed = {info['seed']} zs = {info['redshift']}")
            self.make_save_patches(input_map, output_path)

    def generate_fname(self, data_path, output_dir, ngal = 0):
        fname = data_path.stem + ".npy"
        if ngal == 0:
            fname = fname.replace("kappa", "patches").replace(".npy", f"_oa{int(self.pp.patch_size_deg)}_noiseless.npy")
        else:
            fname = fname.replace("kappa", "patches").replace(".npy", f"_oa{int(self.pp.patch_size_deg)}_ngal{ngal}.npy")
        return output_dir / fname
    
    def load_map(self, data_path):
        logging.debug(f"Loading map from {data_path}")
        map_data = hp.read_map(str(data_path))
        return hp.reorder(map_data, n2r=True)

    def make_save_patches(self, input_map, output_path):
        patches = self.pp.make_patches(input_map)
        np.save(output_path, patches)
        logging.info(f"Saved patches to {output_path}")

def process_file_sl(args: Tuple[Path, int, Path, float, bool]) -> None:
    """
    Process a single combination of data_path and smoothing length (sl).

    Args:
        args (Tuple[Path, int, Path, float, bool]): 
            - data_path: Path to the input .npy file
            - sl: Smoothing length in arcminutes
            - output_dir: Directory to save the processed file
            - patch_size_deg: Patch size in degrees
            - overwrite: Flag to overwrite existing files

    Returns:
        None
    """
    data_path, sl, output_dir, patch_size_deg, overwrite = args
    output_filename = f"{data_path.stem}_sl{sl}.npy"
    output_path = output_dir / output_filename

    if output_path.exists() and not overwrite:
        logging.info(f"Skipping existing file: {output_path}")
        return

    try:
        logging.info(f"Processing {data_path.name} with sl = {sl}")
        data = np.load(data_path)

        # Create and smooth ConvergenceMap objects
        conv_smoothed = np.array([
            ConvergenceMap(slice_data, angle=patch_size_deg * u.deg)
            .smooth(sl * u.arcmin, kind="gaussian")
            .data.astype(np.float32)
            for slice_data in data
        ], dtype=np.float32)

        # Save the smoothed data
        np.save(output_path, conv_smoothed)
        logging.info(f"Saved smoothed data to {output_path.name}")

    except Exception as e:
        logging.error(f"Error processing {data_path.name} with sl={sl}: {e}")

class PatchSmoother:
    def __init__(
        self, 
        data_dir: str, 
        patch_processor: 'PatchProcessor', 
        sl_list: List[int] = [2, 5, 8, 10], 
        overwrite: bool = False,
        num_workers: int = None
    ):
        """
        Initializes the PatchSmoother.

        Args:
            data_dir (str): Directory containing the input .npy files.
            patch_processor (PatchProcessor): Instance of PatchProcessor.
            sl_list (List[int], optional): List of smoothing lengths. Defaults to [2, 5, 8, 10].
            overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.
            num_workers (int, optional): Number of multiprocessing workers. Defaults to number of CPU cores.
        """
        self.data_dir = Path(data_dir)
        self.patch_processor = patch_processor
        self.sl_list = sl_list
        self.overwrite = overwrite

        self.data_paths = sorted(self.data_dir.glob("patch_kappa/*.npy"))
        self.output_dir = self.data_dir / "patch_snr"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = num_workers or mp.cpu_count()

    def run(self) -> None:
        """
        Executes the patch smoothing process using multiprocessing.
        """
        # Prepare arguments for each task
        tasks = [
            (data_path, sl, self.output_dir, self.patch_processor.patch_size_deg, self.overwrite)
            for data_path in self.data_paths
            for sl in self.sl_list
        ]

        logging.info(f"Starting processing with {self.num_workers} workers...")

        # Initialize multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better performance and responsiveness
            for _ in pool.imap_unordered(process_file_sl, tasks):
                pass  # Results are handled within the process_file_sl function

        logging.info("Processing complete.")

if __name__ == "__main__":
    from src.utils.utils import parse_arguments, load_config, filter_config
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    config = load_config(args.config_file)
    filtered_config = filter_config(config, PatchProcessor)
    pp = PatchProcessor(**filtered_config)
    
    pg = PatchGenerator(pp, config["zs_list"], overwrite=args.overwrite)
    pg.run()

    #ps = PatchSmoother(args.datadir, pp, sl_list=[2, 5, 8, 10], overwrite=args.overwrite, num_workers=20)
    #ps.run()