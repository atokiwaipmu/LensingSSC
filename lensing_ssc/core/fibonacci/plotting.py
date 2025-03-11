import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

from lensing_ssc.core.fibonacci.optimizer import PatchOptimizer
from lensing_ssc.core.fibonacci.fibonacci import FibonacciGrid

def plot_coverage_map(coverage_map: np.ndarray, title: str, subplot: tuple, rot: tuple = (0, 0, 0)):
    """
    Creates a Healpy orthview plot of the coverage map.

    Args:
        coverage_map (np.ndarray): The coverage map to display.
        title (str): The title of the plot.
        subplot (tuple): A tuple specifying the subplot layout (nrows, ncols, index).
        rot (tuple, optional): The rotation angles of the view, default is (0, 0, 0).
    """
    hp.orthview(coverage_map, nest=True, half_sky=True, title=title,
                sub=subplot, rot=rot, cmap='viridis', cbar=False)
    hp.graticule()

def plot_fibonacci_grid(optimizer: PatchOptimizer, n: int = None):
    """
    Displays the number of patch pixels along with the Fibonacci grid on a Healpix map.

    Args:
        optimizer (PatchOptimizer): The optimizer to use.
        n (int, optional): The number of patches to plot. If None, N_opt is used.

    Returns:
        tuple: (fig, pixels) where 'pixels' is a list of the number of patch pixels.
    """
    if n is None:
        n = optimizer.N_opt
    if n is None:
        raise ValueError("The optimal number of patches has not been set. Please run optimize() first.")

    npix = hp.nside2npix(optimizer.nside)
    tmp = np.zeros(npix)

    points = FibonacciGrid.fibonacci_grid_on_sphere(n)
    valid_points = points[(points[:, 0] < np.pi - optimizer.radius) & (points[:, 0] > optimizer.radius)]
    invalid_points = points[(points[:, 0] >= np.pi - optimizer.radius) | (points[:, 0] <= optimizer.radius)]

    pixels = []
    for center in valid_points:
        vertices = optimizer.rotated_vertices(center)
        vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
        ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
        tmp[ipix] += 1
        pixels.append(len(ipix))

    for center in invalid_points:
        vertices = optimizer.rotated_vertices(center)
        vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
        ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
        tmp[ipix] -= 1
        pixels.append(len(ipix))

    fig = plt.figure(figsize=(10, 5))

    hp.orthview(tmp, title=f'Fibonacci Grid ({optimizer.patch_size}° Patches), {n} Patches',
                nest=True, half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 1))
    hp.orthview(tmp, title=f'Top View: {n} Patches', nest=True,
                rot=(0, 90, 0), half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 2))

    return fig, pixels