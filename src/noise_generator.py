# src/noise_generator.py

import numpy as np
import healpy as hp
import logging
from typing import Optional
from numpy.random import Generator, PCG64

class NoiseGenerator:
    """
    Generates noise maps for simulating galaxy surveys using the HEALPix pixelization scheme.
    """

    def __init__(self, ngal: int = 30, nside: int = 8192, epsilon: float = 0.3, rng: Optional[Generator] = None) -> None:
        """
        Initializes the NoiseGenerator with configuration parameters.

        Args:
            ngal (int): Number of galaxies per square arcminute.
            nside (int): HEALPix nside parameter determining the map resolution.
            epsilon (float): Intrinsic noise level parameter.
            rng (Optional[Generator]): Instance-specific random number generator.
        """
        self.ngal = ngal
        self.nside = nside
        self.epsilon = epsilon
        self.noise_map: Optional[np.ndarray] = None

        self._validate_parameters()
        self._calculate_properties()

        # Initialize an instance-specific random number generator
        self.rng = rng if rng is not None else Generator(PCG64())
        logging.info(
            f"NoiseGenerator initialized with ngal={self.ngal}, nside={self.nside}, epsilon={self.epsilon}"
        )

    def _validate_parameters(self) -> None:
        """
        Validates the initialization parameters.
        """
        if not isinstance(self.ngal, int) or self.ngal < 0:
            raise ValueError(f"Invalid ngal value: {self.ngal}. It must be a non-negative integer.")
        if not isinstance(self.nside, int) or not hp.isnsideok(self.nside):
            raise ValueError(f"Invalid nside value: {self.nside}. It must be a power of 2.")
        if not isinstance(self.epsilon, (float, int)) or self.epsilon <= 0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. It must be a positive number.")

    def _calculate_properties(self) -> None:
        """
        Calculates internal properties based on initialization parameters.
        """
        try:
            self.pixarea_arcmin2 = hp.nside2pixarea(self.nside, degrees=True) * 60**2  # Convert deg² to arcmin²
            self.npix = hp.nside2npix(self.nside)
            self.sigma = self.epsilon / np.sqrt(self.ngal * self.pixarea_arcmin2)
            logging.debug(
                f"Calculated properties: pixarea_arcmin2={self.pixarea_arcmin2}, "
                f"npix={self.npix}, sigma={self.sigma}"
            )
        except Exception as e:
            logging.error(f"Error in calculating properties: {e}")
            raise

    def generate_noise(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a Gaussian noise map based on the specified seed.

        Args:
            seed (Optional[int]): Random seed for noise generation. If None, randomness is not fixed.

        Returns:
            np.ndarray: The generated noise map.
        """
        if self.noise_map is not None:
            logging.debug("Returning existing noise map.")
            return self.noise_map

        logging.info(f"Generating noise map with seed={seed}")
        if seed is not None:
            self.rng = Generator(PCG64(seed))
            logging.debug(f"Random generator seeded with {seed}")

        try:
            self.noise_map = self.rng.normal(loc=0.0, scale=self.sigma, size=self.npix)
            logging.debug("Noise map generated successfully.")
        except Exception as e:
            logging.error(f"Error in generating noise map: {e}")
            raise

        return self.noise_map

    def save_noise(self, output_path: str, seed: Optional[int] = None) -> None:
        """
        Saves the generated noise map to a FITS file at the specified path.

        Args:
            output_path (str): File path to save the noise map.
            seed (Optional[int]): Random seed for noise generation if noise map is not already generated.
        """
        if self.noise_map is None:
            self.generate_noise(seed=seed)

        try:
            hp.write_map(output_path, self.noise_map, overwrite=True)
            logging.info(f"Noise map saved to '{output_path}'.")
        except Exception as e:
            logging.error(f"Failed to save noise map to '{output_path}': {e}")
            raise

    def add_noise(self, input_map: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Adds the generated noise map to the input map.

        Args:
            input_map (np.ndarray): The input map to which noise will be added.
            seed (Optional[int]): Random seed for noise generation if noise map is not already generated.

        Returns:
            np.ndarray: The input map with noise added.
        """
        if not isinstance(input_map, np.ndarray):
            logging.error("Input map must be a NumPy array.")
            raise TypeError("input_map must be a NumPy array.")

        if input_map.size != self.npix:
            logging.error(
                f"Input map size {input_map.size} does not match expected size {self.npix}."
            )
            raise ValueError(f"Input map size {input_map.size} does not match expected size {self.npix}.")

        noise = self.generate_noise(seed=seed)
        noisy_map = input_map + noise
        logging.debug("Noise added to the input map successfully.")
        return noisy_map

    def reset_noise(self) -> None:
        """
        Resets the noise map, allowing for regeneration.
        """
        self.noise_map = None
        logging.info("Noise map has been reset.")

    def set_ngal(self, ngal: int) -> None:
        """
        Sets the number of galaxies per square arcminute.

        Args:
            ngal (int): Number of galaxies per square arcminute.
        """
        self.ngal = ngal
        self.sigma = self.epsilon / np.sqrt(self.ngal * self.pixarea_arcmin2)
        logging.info(f"ngal set to {self.ngal}.")   

 

if __name__ == "__main__":
    from src.utils import parse_arguments, load_config, filter_config
    from src.info_extractor import InfoExtractor
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG)
    args = parse_arguments()
    config = load_config(args.config_file)

    kappa_dir = Path(args.datadir) / "kappa"
    noisy_dir = Path(args.datadir) / "noisy_maps"
    noisy_dir.mkdir(exist_ok=True, parents=True)

    kappa_map_paths = sorted(kappa_dir.glob("*.fits"))
    ng_config = filter_config(config, NoiseGenerator)
    noise_gen = NoiseGenerator(**ng_config)

    for kappa_map_path in kappa_map_paths:
        info = InfoExtractor.extract_info_from_path(kappa_map_path)

        # make the list of file names planning to be generated
        noisy_paths = []
        for ngal in config.get("ngal_list", []):
            if ngal == 0:
                continue

            noisy_path = noisy_dir / f"{kappa_map_path.stem}_ngal{ngal}.fits"
            noisy_paths.append(noisy_path)
        
        # check if the file already exists, filter out the ones that do
        noisy_paths = [path for path in noisy_paths if not path.exists() or args.overwrite]
        if not noisy_paths:
            logging.info(f"No noisy maps to be generated for {kappa_map_path.name}. Skipping.")
            continue

        logging.info(f"Generating noisy maps for {kappa_map_path.name}.")
        kappa_map = hp.read_map(str(kappa_map_path))
        for noisy_path in noisy_paths:
            ngal = int(noisy_path.stem.split("_")[-1].split("ngal")[-1])
            noise_gen.set_ngal(ngal)
            tmp_seed = int(info["seed"] + ngal + info["redshift"] * 100)
            noisy_map = noise_gen.add_noise(kappa_map, seed=tmp_seed)

            hp.write_map(str(noisy_path), noisy_map, overwrite=args.overwrite, dtype=np.float32)
            logging.info(f"Noisy map saved to {noisy_path.name}")
            noise_gen.reset_noise()