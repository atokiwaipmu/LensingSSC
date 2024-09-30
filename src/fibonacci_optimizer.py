
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from src.fibonacci_helper import FibonacciHelper, SphereRotator

class FibonacciOptimizer:
    """Optimizes the number of patches for a given healpix map.

    Args:
        nside (int): Healpix resolution parameter.
        patch_size (float): Patch size in degrees.
        Ninit (int): Initial number of patches.
        Nstepinit (int): Initial step size for patch count adjustment.
    """

    def __init__(self, nside=1024, patch_size=10, Ninit=280, Nstepinit=10):
        self.nside = nside
        self.patch_size = patch_size
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.Ninit = Ninit
        self.Nstepinit = Nstepinit
        self.N_opt = None

    def optimize(self, verbose=False):
        """Optimizes the patch count.

        Args:
            verbose (bool): Whether to print progress messages.

        Returns:
            int: The optimal number of patches.
        """

        N = self.Ninit
        Nstep = self.Nstepinit

        while Nstep > 0:
            if verbose:
                print("Testing N =", N)

            points = FibonacciHelper.fibonacci_grid_on_sphere(N)
            valid_points = points[(points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)]

            counts = np.zeros(hp.nside2npix(self.nside))
            for center in valid_points:
                vertices = self.rotated_vertices(center)
                vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
                ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
                counts[ipix] += 1

                if np.any(counts[ipix] > 1):
                    N -= Nstep
                    break

            if np.all(counts[ipix] <= 1):
                N += Nstep
                Nstep = Nstep // 2
                if verbose:
                    print("Restarting from N =", N, "with Nstep =", Nstep)

        self.N_opt = N - 1
        return N - 1

    def rotated_vertices(self, center):
        """Rotates the vertices of a patch.

        Args:
            center (tuple): The center of the patch in spherical coordinates.

        Returns:
            numpy.ndarray: The rotated vertices of the patch.
        """

        theta, phi = center
        vertices = self.vertices_from_center([np.pi / 2, 0])
        rotated_vertices = np.array([SphereRotator.rotate_point(vertex, theta, phi) for vertex in vertices])
        return rotated_vertices

    def vertices_from_center(self, center):
        """Calculates the vertices of a patch centered at the given point.

        Args:
            center (tuple): The center of the patch in spherical coordinates.

        Returns:
            numpy.ndarray: The vertices of the patch.
        """

        theta, phi = center
        half_size = np.radians(self.patch_size) / np.sqrt(2)

        vertices = np.array([
            [theta, phi + half_size * np.sin(theta)],
            [theta + half_size, phi],
            [theta, phi - half_size * np.sin(theta)],
            [theta - half_size, phi]
        ])

        return vertices
    
    def plot_fibonacci(self, fig=None, n=None):
        """Plots the Fibonacci grid for a given patch count.

        Args:
            fig (matplotlib.figure.Figure, optional): Figure to plot on.
            n (int, optional): Number of patches to plot.

        Returns:
            tuple: A tuple containing the figure and a list of patch pixel counts.
        """

        if n is None:
            n = self.N_opt

        # Initialize the map
        tmp = np.zeros(hp.nside2npix(self.nside))

        # Generate Fibonacci points and classify them
        points = FibonacciHelper.fibonacci_grid_on_sphere(n)
        valid_points = points[(points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)]
        invalid_points = points[(points[:, 0] >= np.pi - self.radius) | (points[:, 0] <= self.radius)]

        # Calculate patch pixel counts for valid and invalid points
        pixels = []
        for center in valid_points:
            vertices = self.rotated_vertices(center)
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
            tmp[ipix] += 1
            pixels.append(len(ipix))

        for center in invalid_points:
            vertices = self.rotated_vertices(center)
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
            tmp[ipix] -= 1
            pixels.append(len(ipix))

        # Create a new figure if not provided
        if fig is None:
            fig = plt.figure(figsize=(10, 5))

        # Plot the Fibonacci grid
        hp.orthview(tmp, title=f'Fibonacci grid ({self.patch_size}x{self.patch_size} '+r'$deg^2$)'+f', {n} Patch', nest=True, half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 1))
        hp.orthview(tmp, title=f'Fibonacci grid ({self.patch_size}x{self.patch_size} '+r'$deg^2$)'+f', {n} Patch: Top View', nest=True, rot=(0, 90, 0), half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 2))

        return fig, pixels
