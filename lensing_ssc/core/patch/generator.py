import logging
from pathlib import Path
from typing import List

import healpy as hp
import numpy as np

from lensing_ssc.core.patch.processor import PatchProcessor
from lensing_ssc.utils.extractors import InfoExtractor


class PatchGenerator:
    """
    Generates patches from full-sky kappa maps using a provided PatchProcessor.

    Parameters
    ----------
    pp : PatchProcessor
        An instance of PatchProcessor.
    zs_list : List[float]
        List of source redshifts to process.
    data_dir : str, optional
        Directory containing full-sky maps. Defaults to the specified path.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    """
    def __init__(
        self,
        pp: PatchProcessor,
        zs_list: List[float],
        data_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/data",
        overwrite: bool = False
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.zs_list: List[float] = zs_list
        self.pp: PatchProcessor = pp
        self.overwrite: bool = overwrite

        # Expect files organized under fullsky/<seed>/zs*/ with filenames matching kappa_zs*_s*.fits
        self.kappa_paths: List[Path] = sorted(self.data_dir.glob("fullsky/*/zs*/*.fits"))
        self.output_dir: Path = self.data_dir / f"patch_oa{int(self.pp.patch_size_deg)}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dir_initialize()

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
        Process each full-sky kappa map to generate patches.
        """
        for kappa_path in self.kappa_paths:
            info = InfoExtractor.extract_info_from_path(kappa_path)
            box_type: str = kappa_path.parts[-3]  # Assumes structure: .../fullsky/<seed>/zs*/...
            input_map: np.ndarray = self.load_map(kappa_path)
            output_dir: Path = self.output_dir / box_type / f"zs{info['redshift']}"
            logging.info(f"Output directory: {output_dir}")
            output_path: Path = self.generate_fname(kappa_path, output_dir)
            if output_path.exists() and not self.overwrite:
                logging.info(f"Skipping existing file: {output_path}")
                continue
            logging.info(f"Generating patches for seed = {info['seed']} zs = {info['redshift']}")
            self.make_save_patches(input_map, output_path)

    def generate_fname(self, data_path: Path, output_dir: Path, ngal: int = 0) -> Path:
        """
        Generate the output filename based on the input file and noise level.

        Parameters
        ----------
        data_path : Path
            Path to the input kappa map file.
        output_dir : Path
            Output directory to save patches.
        ngal : int, optional
            Galaxy noise parameter. Defaults to 0.

        Returns
        -------
        Path
            Generated output file path.
        """
        fname: str = data_path.stem + ".npy"
        if ngal == 0:
            fname = fname.replace("kappa", "patches").replace(
                ".npy", f"_oa{int(self.pp.patch_size_deg)}_noiseless.npy"
            )
        else:
            fname = fname.replace("kappa", "patches").replace(
                ".npy", f"_oa{int(self.pp.patch_size_deg)}_ngal{ngal}.npy"
            )
        return output_dir / fname

    def load_map(self, data_path: Path) -> np.ndarray:
        """
        Load and reorder a Healpy map from the given file.

        Parameters
        ----------
        data_path : Path
            Path to the .fits file.

        Returns
        -------
        np.ndarray
            Reordered kappa map.
        """
        logging.debug(f"Loading map from {data_path}")
        map_data: np.ndarray = hp.read_map(str(data_path))
        return hp.reorder(map_data, n2r=True)

    def make_save_patches(self, input_map: np.ndarray, output_path: Path) -> None:
        """
        Extract patches using the PatchProcessor and save them to disk.

        Parameters
        ----------
        input_map : np.ndarray
            The full-sky kappa map.
        output_path : Path
            File path where the patches are saved.
        """
        patches: np.ndarray = self.pp.make_patches(input_map)
        np.save(output_path, patches)
        logging.info(f"Saved patches to {output_path}")