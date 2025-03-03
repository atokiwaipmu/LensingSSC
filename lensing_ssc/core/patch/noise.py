import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from lensing_ssc.core.patch.processor import PatchProcessor
from lensing_ssc.utils.extractors import InfoExtractor


class NoiseAdder:
    """
    Adds shape noise to convergence maps based on galaxy density.

    Parameters
    ----------
    epsilon : float
        Shape noise amplitude parameter, typically around 0.3.
    ngal_list : List[int]
        List of galaxy densities (galaxies/arcmin^2) to process.
    data_dir : str, optional
        Directory containing noiseless patches. Defaults to the specified path.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    """
    def __init__(
        self,
        epsilon: float,
        ngal_list: List[int],
        data_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/data",
        overwrite: bool = False
    ) -> None:
        self.epsilon: float = epsilon
        self.ngal_list: List[int] = ngal_list
        self.data_dir: Path = Path(data_dir)
        self.overwrite: bool = overwrite
        
        # Find patch directories
        self.patch_dirs = sorted(self.data_dir.glob("patches/*/zs*"))
        self.dir_initialize()
        
    def dir_initialize(self) -> None:
        """
        Create necessary output subdirectories for each noise level.
        """
        for patch_dir in self.patch_dirs:
            for ngal in self.ngal_list:
                output_subdir: Path = patch_dir / f"ngal{ngal}"
                output_subdir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory {output_subdir}")

    def run(self) -> None:
        """
        Process each noiseless patch file to generate noisy versions.
        """
        # Find all noiseless patch files
        noiseless_files = []
        for patch_dir in self.patch_dirs:
            noiseless_files.extend(sorted((patch_dir / "noiseless").glob("*.npy")))
        
        for noiseless_file in noiseless_files:
            # Load patches
            patches = np.load(noiseless_file)
            
            # Extract patch parameters
            filename = noiseless_file.name
            info = InfoExtractor.extract_info_from_path(noiseless_file)
            
            # Get patch size from filename (assuming filename format like patches_zs*_s*_oa10_noiseless.npy)
            patch_size_deg = float(filename.split("_oa")[1].split("_")[0])
            patch_size_arcmin = patch_size_deg * 60
            
            # Process dimension info
            patch_shape = patches.shape
            if len(patch_shape) == 4:  # Assumes format [n_patches, 1, height, width]
                npix = patch_shape[2] * patch_shape[3]
            else:  # Assumes format [n_patches, height, width]
                npix = patch_shape[1] * patch_shape[2]
                
            # Calculate pixel area in arcmin^2
            pixarea_arcmin2 = (patch_size_arcmin ** 2) / npix
            
            # Process each noise level
            for ngal in self.ngal_list:
                # Generate output path
                output_dir = noiseless_file.parent.parent / f"ngal{ngal}"
                output_path = output_dir / filename.replace("noiseless", f"ngal{ngal}")
                
                if output_path.exists() and not self.overwrite:
                    logging.info(f"Skipping existing file: {output_path}")
                    continue
                
                logging.info(f"Adding noise (ngal={ngal}) to {noiseless_file}")
                
                # Add noise
                noisy_patches = self.add_noise(patches, ngal, npix, pixarea_arcmin2)
                
                # Save noisy patches
                np.save(output_path, noisy_patches.astype(np.float32))
                logging.info(f"Saved noisy patches to {output_path}")
    
    def add_noise(self, patches: np.ndarray, ngal: int, npix: int, pixarea_arcmin2: float) -> np.ndarray:
        """
        Add shape noise to patches based on galaxy density.
        
        Parameters
        ----------
        patches : np.ndarray
            Noiseless convergence patches.
        ngal : int
            Galaxy density (galaxies/arcmin^2).
        npix : int
            Total number of pixels in each patch.
        pixarea_arcmin2 : float
            Area of each pixel in arcmin^2.
            
        Returns
        -------
        np.ndarray
            Noisy convergence patches.
        """
        # Generate noise with standard deviation based on galaxy density
        noise_std = self.epsilon / np.sqrt(ngal * pixarea_arcmin2)
        
        # Handle different patch array shapes
        if len(patches.shape) == 4:  # [n_patches, 1, height, width]
            noise = np.random.normal(0, noise_std, patches.shape)
        else:  # [n_patches, height, width]
            noise = np.random.normal(0, noise_std, patches.shape)
            
        return patches + noise


def add_noise_to_patches(
    epsilon: float = 0.26, 
    ngal_list: List[int] = [7, 15, 30, 50], 
    data_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/data/patches",
    overwrite: bool = False
) -> None:
    """
    Convenience function to add noise to patches.
    
    Parameters
    ----------
    epsilon : float, optional
        Shape noise amplitude parameter. Defaults to 0.26.
    ngal_list : List[int], optional
        List of galaxy densities to process. Defaults to [7, 15, 30, 50].
    data_dir : str, optional
        Directory containing noiseless patches. Defaults to the specified path.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    """
    noise_adder = NoiseAdder(
        epsilon=epsilon,
        ngal_list=ngal_list,
        data_dir=data_dir,
        overwrite=overwrite
    )
    noise_adder.run()