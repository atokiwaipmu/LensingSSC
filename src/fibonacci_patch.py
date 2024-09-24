
import numpy as np
import healpy as hp
from scipy.ndimage import rotate
import math
import matplotlib.pyplot as plt

class PatchOptimizer:
    def __init__(self, nside=1024, patch_size=10, Ninit=280, Nstepinit=10):
        self.nside = nside
        self.patch_size = patch_size
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.Ninit = Ninit
        self.Nstepinit = Nstepinit
        self.N_opt = None

    def optimize(self, verbose=False):
        N = self.Ninit
        Nstep = self.Nstepinit

        while Nstep > 0:
            if verbose: print("testing N = ", N)
            points = PatchOptimizer.fibonacci_grid_on_sphere(N)
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
                if verbose: print("Restarting from N = {}, w/ Nstep = {}".format(N, Nstep)) 

        self.N_opt = N - 1
        return N - 1

    def rotated_vertices(self, center):
        theta, phi = center
        vertices = self.vertices_from_center([np.pi/2, 0])
        rotated_vertices = np.array([PatchOptimizer.rotate_point(vertex, theta, phi) for vertex in vertices])
        return rotated_vertices
    
    def vertices_from_center(self, center):
        theta, phi = center
        half_size = np.radians(self.patch_size) / np.sqrt(2)
        
        vertices = np.array([
            [theta, phi + half_size * np.sin(theta)],
            [theta + half_size, phi],
            [theta, phi - half_size  * np.sin(theta)],
            [theta - half_size, phi]
        ])
        
        return vertices   
    
    def plot_fibonacci(self, fig=None, n=None):
        if n is None:
            n = self.N_opt
        tmp = np.zeros(hp.nside2npix(self.nside))
        points = PatchOptimizer.fibonacci_grid_on_sphere(n)
        valid_points = points[(points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)]
        invalid_points = points[(points[:, 0] > np.pi - self.radius) | (points[:, 0] < self.radius)]

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

        if fig is None:
            fig = plt.figure(figsize=(10, 5))

        # Plot
        hp.orthview(tmp, title=f'Fibonacci grid ({self.patch_size}x{self.patch_size} '+r'$deg^2$)'+f', {n} Patch', nest=True, half_sky=True, cbar=False, cmap='viridis', fig = fig, sub=(1, 2, 1))
        hp.orthview(tmp, title=f'Fibonacci grid ({self.patch_size}x{self.patch_size} '+r'$deg^2$)'+f', {n} Patch: Top View', nest=True, rot=(0, 90, 0), half_sky=True, cbar=False, cmap='viridis', fig = fig, sub=(1, 2, 2))

        return fig, pixels

    @staticmethod
    def M(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    @staticmethod
    def spherical_to_cartesian(point, r=1):
        theta, phi = point
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def rotation_matrix(theta, phi):
        """
        Generate a rotation matrix for rotating a point on the sphere by theta and phi.
        """
        R_phi = PatchOptimizer.M([0, 0, 1], phi)
        R_theta = PatchOptimizer.M(np.array([np.sin(phi), -np.cos(phi), 0]), np.pi/2 - theta)
        return np.dot(R_theta, R_phi)

    @staticmethod
    def rotate_point(point, theta, phi):
        """
        Rotate a point on the sphere by theta and phi.
        """
        x, y, z = PatchOptimizer.spherical_to_cartesian(point)
        x_rot, y_rot, z_rot = np.dot(PatchOptimizer.rotation_matrix(theta, phi), np.array([x, y, z]))
        return PatchOptimizer.cartesian_to_spherical(x_rot, y_rot, z_rot)[1:]
    
    @staticmethod
    def fibonacci_grid_on_sphere(N):
        points = np.zeros((N, 2))
        
        phi = (np.sqrt(5) + 1) / 2  # Golden ratio
        golden_angle = 2 * np.pi / phi
        
        for i in range(N):
            theta = np.arccos(1 - 2 * (i + 0.5) / N)
            phi = (golden_angle * i) % (2 * np.pi)
            points[i] = [theta, phi]
        
        return points

    @staticmethod
    # Function to get pixel values inside a square patch
    def get_patch_pixels(image, side_length):
        rotated_image = rotate(image, 45, reshape=False)

        x_center, y_center = rotated_image.shape[1]//2 , rotated_image.shape[0]//2
        half_side = side_length // 2

        x_start = max(x_center - half_side, 0)
        x_end = min(x_center + half_side, rotated_image.shape[1])
        
        y_start = max(y_center - half_side, 0)
        y_end = min(y_center + half_side, rotated_image.shape[0])
        
        patch_pixels = rotated_image[y_start:y_end, x_start:x_end]
        
        return patch_pixels