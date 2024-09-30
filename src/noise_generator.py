

import logging
import numpy as np
import healpy as hp
from typing import Optional

class NoiseGenerator:
    """
    Generates noise maps for simulating galaxy surveys.
    """

    def __init__(self, ngal: int = 30, nside: int = 8192, epsilon: float = 0.3) -> None:
        """
        Initializes the NoiseGenerator with configuration parameters.

        Args:
            ngal: Number of galaxies per square arcmin.
            nside: HEALPix nside parameter for the map.
            epsilon: Noise level parameter.
        """
        self.ngal = ngal
        self.nside = nside
        self.epsilon = epsilon

        self._calculate_properties()
        logging.info(f"NoiseGenerator initialized: ngal={self.ngal}, nside={self.nside}, epsilon={self.epsilon}")

    def _calculate_properties(self):
        """Calculates internal properties based on configuration."""
        self.pixarea = hp.nside2pixarea(self.nside, degrees=True) * 60 ** 2  # arcmin^2
        self.npix = hp.nside2npix(self.nside)
        self.sigma = self.epsilon / np.sqrt(self.ngal * self.pixarea)
        self.noise_map = None

    def generate_noise(self, seed: Optional[int] = 0) -> np.ndarray:
        """
        Generates a noise map with the specified seed.

        Args:
            seed: Random seed for noise generation (defaults to 0).

        Returns:
            The generated noise map.
        """
        if self.noise_map is None:
            logging.info(f"Generating noise with seed {seed}")
            np.random.seed(seed)
            self.noise_map = np.random.normal(loc=0, scale=self.sigma, size=(self.npix,))
        return self.noise_map

    def save_noise(self, output_path: str, seed: Optional[int] = 0) -> None:
        """
        Saves the generated noise map to the specified path.

        Args:
            output_path: Path to save the noise map.
        """
        if self.noise_map is None:
            self.noise_map = self.generate_noise(seed=seed)
        hp.write_map(output_path, self.noise_map)
        logging.info(f"Noise map saved to: {output_path}")

    def add_noise(self, input_map: np.ndarray, seed: Optional[int] = 0) -> np.ndarray:
        """
        Adds the generated noise map to the input map.

        Args:
            input_map: The input map to add noise to.

        Returns:
            The input map with noise added.
        """
        if self.noise_map is None:
            self.noise_map = self.generate_noise(seed=seed)
        return input_map + self.noise_map