import numpy as np
import healpy as hp
import logging
from typing import Tuple, List

from lensing_ssc.core.fibonacci.optimizer import PatchOptimizer


class CoverageAnalyzer:
    def build_mask_first_scenario(points: np.ndarray, optimizer: PatchOptimizer) -> np.ndarray:
        """
        Build a boolean mask for scenario 1 by excluding points near the poles.

        Parameters
        ----------
        points : np.ndarray
            Array of (theta, phi) points.
        optimizer : FibonacciOptimizer
            Provides the patch radius.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates an invalid point.
        """
        mask_northpole = points[:, 0] <= optimizer.radius
        mask_southpole = points[:, 0] >= np.pi - optimizer.radius
        combined_mask = mask_northpole | mask_southpole
        return combined_mask

    def build_mask_second_scenario(points: np.ndarray, optimizer: PatchOptimizer) -> np.ndarray:
        """
        Build a more complex mask excluding pole-adjacent and certain equatorial/longitudinal zones.

        Parameters
        ----------
        points : np.ndarray
            Array of (theta, phi) points.
        optimizer : FibonacciOptimizer
            Provides the patch radius.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates an invalid point.
        """
        mask_northpole = points[:, 0] <= optimizer.radius
        mask_southpole = points[:, 0] >= np.pi - optimizer.radius

        equator_threshold = optimizer.radius / 2
        longitude_threshold = optimizer.radius / 2

        mask_equator = np.abs(points[:, 0] - np.pi/2) <= equator_threshold
        mask_longitude1 = np.abs(points[:, 1]) <= longitude_threshold
        mask_longitude2 = np.abs(points[:, 1] - np.pi/2) <= longitude_threshold
        mask_longitude3 = np.abs(points[:, 1] - np.pi) <= longitude_threshold
        mask_longitude4 = np.abs(points[:, 1] - 3*np.pi/2) <= longitude_threshold
        mask_longitude5 = np.abs(points[:, 1] - 2*np.pi) <= longitude_threshold

        combined_mask = (mask_northpole | mask_southpole |
                        mask_equator | mask_longitude1 |
                        mask_longitude2 | mask_longitude3 |
                        mask_longitude4 | mask_longitude5)
        return combined_mask

    def fill_coverage_map(valid_points: np.ndarray, invalid_points: np.ndarray,
                        optimizer: PatchOptimizer, coverage_map: np.ndarray) -> None:
        """
        Fill a coverage map by incrementing for valid patches and decrementing for invalid ones.

        Parameters
        ----------
        valid_points : np.ndarray
            Spherical coordinates of valid points.
        invalid_points : np.ndarray
            Spherical coordinates of invalid points.
        optimizer : FibonacciOptimizer
            Provides methods for patch vertex computation and nside.
        coverage_map : np.ndarray
            Array to be modified in place with coverage data.
        """
        for center in valid_points:
            vertices = optimizer.rotated_vertices(center)
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
            coverage_map[ipix] += 1

        for center in invalid_points:
            vertices = optimizer.rotated_vertices(center)
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
            coverage_map[ipix] -= 1

    def compute_coverage_stats(coverage_map: np.ndarray) -> float:
        """
        Compute the fraction of sky coverage from a coverage map.

        Parameters
        ----------
        coverage_map : np.ndarray
            Map where each pixel's integer value indicates coverage.

        Returns
        -------
        float
            Fraction of pixels with a value of 1 relative to total (covered or uncovered).
        """
        area_covered = np.sum(coverage_map == 1)
        area_not_covered = np.sum(coverage_map == 0) + np.sum(coverage_map == -1)
        total = area_covered + area_not_covered
        if total == 0:
            logging.warning("No valid pixels found in coverage map, returning 0.")
            return 0.0
        return area_covered / total