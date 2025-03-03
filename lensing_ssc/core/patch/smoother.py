import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np
from lenstools import ConvergenceMap
from astropy import units as u

from lensing_ssc.core.patch.processor import PatchProcessor


def process_file_sl(args: Tuple[Path, int, Path, float, bool]) -> None:
    """
    Process a single file for a given smoothing length.

    Parameters
    ----------
    args : Tuple[Path, int, Path, float, bool]
        Contains:
            - data_path: Path to the input .npy file.
            - sl: Smoothing length in arcminutes.
            - output_dir: Directory to save the processed file.
            - patch_size_deg: Patch size in degrees.
            - overwrite: Flag to overwrite existing files.
    """
    data_path, sl, output_dir, patch_size_deg, overwrite = args
    output_filename: str = f"{data_path.stem}_sl{sl}.npy"
    output_path: Path = output_dir / output_filename

    if output_path.exists() and not overwrite:
        logging.info(f"Skipping existing file: {output_path}")
        return

    try:
        logging.info(f"Processing {data_path.name} with sl = {sl}")
        data: np.ndarray = np.load(data_path)

        conv_smoothed: np.ndarray = np.array([
            ConvergenceMap(slice_data, angle=patch_size_deg * u.deg)
            .smooth(sl * u.arcmin, kind="gaussian")
            .data.astype(np.float32)
            for slice_data in data
        ], dtype=np.float32)

        np.save(output_path, conv_smoothed)
        logging.info(f"Saved smoothed data to {output_path.name}")
    except Exception as e:
        logging.error(f"Error processing {data_path.name} with sl={sl}: {e}")

class PatchSmoother:
    """
    Applies Gaussian smoothing to patch files and saves the processed outputs.

    Parameters
    ----------
    data_dir : str
        Directory containing input .npy patch files.
    patch_processor : PatchProcessor
        Instance of PatchProcessor.
    sl_list : List[int], optional
        List of smoothing lengths in arcminutes. Defaults to [2, 5, 8, 10].
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    num_workers : int, optional
        Number of multiprocessing workers. Defaults to CPU count.
    """
    def __init__(
        self,
        data_dir: str,
        patch_processor: PatchProcessor,
        sl_list: List[int] = [2, 5, 8, 10],
        overwrite: bool = False,
        num_workers: int = None
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.patch_processor: PatchProcessor = patch_processor
        self.sl_list: List[int] = sl_list
        self.overwrite: bool = overwrite

        self.data_paths: List[Path] = sorted(self.data_dir.glob("patch_kappa/*.npy"))
        self.output_dir: Path = self.data_dir / "smoothed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers: int = num_workers or mp.cpu_count()

    def dir_initialize(self) -> None:
        """
        Create necessary output subdirectories for each redshift and box type.
        """
        for zs in self.zs_list:
            for box_type in ["bigbox", "tiled"]:
                output_subdir: Path = self.output_dir / box_type / f"zs{zs}"
                output_subdir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory {output_subdir}")

    def run(self) -> None:
        """
        Run the patch smoothing process using multiprocessing.
        """
        tasks: List[Tuple[Path, int, Path, float, bool]] = [
            (data_path, sl, self.output_dir, self.patch_processor.patch_size_deg, self.overwrite)
            for data_path in self.data_paths
            for sl in self.sl_list
        ]
        logging.info(f"Starting processing with {self.num_workers} workers...")
        with mp.Pool(processes=self.num_workers) as pool:
            for _ in pool.imap_unordered(process_file_sl, tasks):
                pass
        logging.info("Processing complete.")