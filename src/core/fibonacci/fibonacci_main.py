import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import logging

from src.core.fibonacci.fibonacci_helper import FibonacciHelper
from src.core.fibonacci.fibonacci_optimizer import FibonacciOptimizer

def initialize_optimizer(nside: int, patch_size: int, n_init: int = 2048, data_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/center_points/"):
    """
    Initialize and optimize the Fibonacci optimizer.

    Parameters
    ----------
    nside : int
        Healpy NSIDE parameter.
    patch_size : int
        Patch size for the optimizer.
    n_init : int, optional
        Initial N parameter for optimization, by default 2048.
    n_step_init : int, optional
        Step for adjusting N during optimization, by default 10.

    Returns
    -------
    FibonacciOptimizer
        An initialized and optimized FibonacciOptimizer instance.
    """
    optimizer = FibonacciOptimizer(nside, patch_size, Ninit=n_init)

    file_path = data_dir + f"fibonacci_points_{optimizer.patch_size}.txt"
    try:
        points = np.loadtxt(file_path)
        optimizer.N_opt = len(points)
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}. Optimizing new points...")

        # Optimize the optimizer
        optimizer.optimize(verbose=True)

    return optimizer


def generate_fibonacci_points(optimizer, data_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/center_points/") -> np.ndarray:
    """
    Generate Fibonacci grid points on a sphere.

    Parameters
    ----------
    optimizer : FibonacciOptimizer
        The optimizer that holds the number of points (N_opt) and radius.

    Returns
    -------
    np.ndarray
        Array of spherical coordinates (theta, phi) for each point.
    """
    file_path = data_dir + f"fibonacci_points_{optimizer.patch_size}.txt"
    try:
        return np.loadtxt(file_path)
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}. Generating new points...")

    points = FibonacciHelper.fibonacci_grid_on_sphere(optimizer.N_opt)
    np.savetxt(file_path, points)
    return points


def build_mask_first_scenario(points: np.ndarray, optimizer) -> np.ndarray:
    """
    Build a boolean mask for the first scenario (north pole, south pole).
    Excludes points close to the poles.

    Parameters
    ----------
    points : np.ndarray
        Array of (theta, phi) points on the sphere.
    optimizer : FibonacciOptimizer
        Optimizer that provides the `radius` and `nside`.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an invalid point (to be excluded).
    """
    mask_northpole = points[:, 0] <= optimizer.radius
    mask_southpole = points[:, 0] >= np.pi - optimizer.radius

    # Combine masks to exclude these pole-adjacent points
    combined_mask = mask_northpole | mask_southpole
    return combined_mask


def build_mask_second_scenario(points: np.ndarray, optimizer) -> np.ndarray:
    """
    Build a more complex mask that excludes regions near the poles
    and certain equatorial/longitudinal zones.

    Parameters
    ----------
    points : np.ndarray
        Array of (theta, phi) points on the sphere.
    optimizer : FibonacciOptimizer
        Optimizer that provides `radius` and `nside`.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an invalid point (to be excluded).
    """
    mask_northpole = points[:, 0] <= optimizer.radius
    mask_southpole = points[:, 0] >= np.pi - optimizer.radius

    equator_threshold = optimizer.radius / 2
    longitude_threshold = optimizer.radius / 2

    # Exclude points near the equator
    mask_equator = np.abs(points[:, 0] - np.pi/2) <= equator_threshold

    # Exclude points near key longitudes
    mask_longitude1 = np.abs(points[:, 1]) <= longitude_threshold
    mask_longitude2 = np.abs(points[:, 1] - np.pi/2) <= longitude_threshold
    mask_longitude3 = np.abs(points[:, 1] - np.pi) <= longitude_threshold
    mask_longitude4 = np.abs(points[:, 1] - 3*np.pi/2) <= longitude_threshold
    mask_longitude5 = np.abs(points[:, 1] - 2*np.pi) <= longitude_threshold

    # Combine all exclusion masks
    combined_mask = (
        mask_northpole | mask_southpole |
        mask_equator | mask_longitude1 |
        mask_longitude2 | mask_longitude3 |
        mask_longitude4 | mask_longitude5
    )
    return combined_mask


def fill_coverage_map(
    valid_points: np.ndarray,
    invalid_points: np.ndarray,
    optimizer,
    coverage_map: np.ndarray
) -> None:
    """
    Fill a coverage map by adding +1 for patches around valid points
    and -1 for patches around invalid points.

    Parameters
    ----------
    valid_points : np.ndarray
        Spherical coordinates of valid points.
    invalid_points : np.ndarray
        Spherical coordinates of invalid (excluded) points.
    optimizer : FibonacciOptimizer
        Provides methods like `rotated_vertices` and the `nside`.
    coverage_map : np.ndarray
        Array to fill with coverage data (modified in place).
    """
    # Mark coverage for valid points
    for center in valid_points:
        vertices = optimizer.rotated_vertices(center)
        vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
        ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
        coverage_map[ipix] += 1

    # Mark coverage for invalid points
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
        Map where each pixel is an integer representing coverage (+1, -1, 0, etc.).

    Returns
    -------
    float
        Fraction of covered pixels relative to the total number of pixels that are
        either covered or uncovered.
    """
    area_covered = np.sum(coverage_map == 1)
    area_not_covered = np.sum(coverage_map == 0) + np.sum(coverage_map == -1)
    if (area_covered + area_not_covered) == 0:
        logging.warning("No coverage or uncovered pixels found, returning 0.")
        return 0.0
    return area_covered / (area_covered + area_not_covered)


def plot_coverage_map(coverage_map: np.ndarray, title: str, subplot: tuple, rot: tuple = (0, 0, 0)):
    """
    Helper function to plot a Healpy orthview of a coverage map.

    Parameters
    ----------
    coverage_map : np.ndarray
        The coverage map to display.
    title : str
        Title for the plot.
    subplot : tuple
        A (nrows, ncols, index) tuple for the orthview subplot position.
    rot : tuple, optional
        Rotation angles for the orthview, by default (0, 0, 0).
    """
    hp.orthview(
        coverage_map,
        nest=True,
        half_sky=True,
        title=title,
        sub=subplot,
        rot=rot,
        cbar=False
    )
    hp.graticule()


def run_first_scenario(optimizer, save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/plot/" , points = None):
    """
    Demonstration of the first scenario, where only the north and south
    pole regions are excluded from valid points.
    """
    if points is None:
        points = generate_fibonacci_points(optimizer)
    mask = build_mask_first_scenario(points, optimizer)

    coverage_map = np.zeros(hp.nside2npix(optimizer.nside))
    valid_points = points[~mask]
    invalid_points = points[mask]

    fill_coverage_map(valid_points, invalid_points, optimizer, coverage_map)
    coverage_fraction = compute_coverage_stats(coverage_map)
    print(f"Percent covered (scenario 1): {coverage_fraction:.2f}")

    # Plot results
    fig = plt.figure(figsize=(10, 5))
    plot_coverage_map(coverage_map, "Front View", (1, 2, 1))
    plot_coverage_map(coverage_map, "Top View", (1, 2, 2), rot=(0, 90, 0))

    # Save the plot
    plt.savefig(save_dir + f"coverage_map_{optimizer.nside}_{optimizer.patch_size}.png", bbox_inches='tight')
    #plt.show()


def run_second_scenario(optimizer, save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/plot/" , points = None):
    """
    Demonstration of the second scenario with additional equatorial and
    longitude-based exclusions.
    """
    if points is None:
        points = generate_fibonacci_points(optimizer)
    mask = build_mask_second_scenario(points, optimizer)

    coverage_map = np.zeros(hp.nside2npix(optimizer.nside))
    valid_points = points[~mask]
    invalid_points = points[mask]

    fill_coverage_map(valid_points, invalid_points, optimizer, coverage_map)
    coverage_fraction = compute_coverage_stats(coverage_map)
    print(f"Percent covered (scenario 2): {coverage_fraction:.2f}")

    # Plot results
    fig = plt.figure(figsize=(15, 5))
    # Subplot 1
    plot_coverage_map(coverage_map, "Front View", (1, 3, 1))
    # Subplot 2
    plot_coverage_map(coverage_map, "Top View (Pole Overlap Excluded)", (1, 3, 2), rot=(0, 90, 0))
    # Subplot 3
    plot_coverage_map(coverage_map, "Front View (Heavily Repeated Region Excluded)", (1, 3, 3))

    # Save the plot
    plt.savefig(save_dir + f"coverage_map_{optimizer.nside}_{optimizer.patch_size}_scenario2.png", bbox_inches='tight')
    #plt.show()

    # Optionally, a mollview as well
    #hp.mollview(coverage_map, nest=True, title="Mollview Coverage", cbar=False)
    #plt.show()


def main(patch_size = 5, nside = 8192):
    """
    Main execution method to demonstrate both coverage scenarios.
    Adjust nside and patch_size as needed.
    """
    optimizer = initialize_optimizer(nside, patch_size)
    points = generate_fibonacci_points(optimizer)

    # Run first scenario
    run_first_scenario(optimizer, points=points)

    # Run second scenario
    run_second_scenario(optimizer, points=points)


if __name__ == "__main__":
    main(patch_size=5)
    main(patch_size=10)
    main(patch_size=15)
    main(patch_size=20)
    main(patch_size=25)
    main(patch_size=30)
