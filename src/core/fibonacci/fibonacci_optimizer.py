
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from src.utils.fibonacci_helper import FibonacciHelper, SphereRotator

class FibonacciOptimizer:
    """Optimizes the number of patches for a given healpix map.

    Args:
        nside (int): Healpix resolution parameter.
        patch_size (float): Patch size in degrees.
        Ninit (int): Initial number of patches.
        Nstepinit (int): Initial step size for patch count adjustment.
    """

    def __init__(self, nside=1024, patch_size=10, Ninit=280):
        self.nside = nside
        self.patch_size = patch_size
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.Ninit = Ninit
        self.N_opt = None

    def optimize(self, verbose=False, read=True):
        """
        Optimizes the patch count using a binary search approach.

        Returns:
            int: The optimal number of patches (N_opt).
        """
        # 1. Define search bounds
        low = 1
        high = self.Ninit
        best_feasible = 0

        # 2. Binary search
        while low <= high:
            mid = (low + high) // 2
            if verbose:
                print(f"Testing feasibility for N={mid} ...")

            if self.is_feasible(mid):
                # If mid is feasible, record it and try bigger
                best_feasible = mid
                low = mid + 1
                if verbose:
                    print(f"N={mid} is feasible, trying larger.")
            else:
                # If mid is not feasible, try smaller
                high = mid - 1
                if verbose:
                    print(f"N={mid} is not feasible, trying smaller.")

        # 3. Save and return best feasible solution
        self.N_opt = best_feasible
        return best_feasible
    
    def is_feasible(self, N):
        """
        Returns True if we can place patches at N points without overlap.
        False otherwise.
        """
        # 1) Generate N points on sphere
        points = FibonacciHelper.fibonacci_grid_on_sphere(N)

        # 2) Filter points by radius constraints
        #    (assuming points[:, 0] is theta, the polar angle)
        valid_points = points[
            (points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)
        ]

        # 3) Build counts array for each pixel
        counts = np.zeros(hp.nside2npix(self.nside), dtype=int)

        for center in valid_points:
            # 3a) Rotate patch corners relative to this center
            vertices = self.rotated_vertices(center)

            # 3b) Convert (theta, phi) vertices to unit vectors
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])

            # 3c) Query all pixels in polygon
            ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)

            # 3d) Increment coverage count
            counts[ipix] += 1

            # 3e) Early break if overlap
            if np.any(counts[ipix] > 1):
                return False

        return True

    def rotated_vertices(self, center):
        """Rotates the vertices of a patch.

        Args:
            center (tuple): The center of the patch in spherical coordinates.

        Returns:
            numpy.ndarray: The rotated vertices of the patch.
        """

        theta, phi = center
        vertices = self.vertices_from_center([np.pi / 2, 0])
        rotated_vertices = np.array([SphereRotator.rotate_point(theta, phi, vertex) for vertex in vertices])
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
