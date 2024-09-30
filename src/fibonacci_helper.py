
import math
import numpy as np
from scipy.ndimage import rotate

class Rotation:
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
    
class SphericalConverter:
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
    
class SphereRotator:
    @staticmethod
    def rotation_matrix(theta, phi):
        """
        Generate a rotation matrix for rotating a point on the sphere by theta and phi.
        """
        R_phi = Rotation.M([0, 0, 1], phi)
        R_theta = Rotation.M(np.array([np.sin(phi), -np.cos(phi), 0]), np.pi/2 - theta)
        return np.dot(R_theta, R_phi)

    @staticmethod
    def rotate_point(point, theta, phi):
        """
        Rotate a point on the sphere by theta and phi.
        """
        x, y, z = SphericalConverter.spherical_to_cartesian(point)
        x_rot, y_rot, z_rot = np.dot(SphereRotator.rotation_matrix(theta, phi), np.array([x, y, z]))
        return SphericalConverter.cartesian_to_spherical(x_rot, y_rot, z_rot)[1:]

class FibonacciHelper:
    @staticmethod
    def fibonacci_grid_on_sphere(N):
        """
        Generates a grid of N points distributed over the surface of a sphere using 
        the Fibonacci lattice method.
        """
        points = np.zeros((N, 2))
        phi = (np.sqrt(5) + 1) / 2  # Golden ratio
        golden_angle = 2 * np.pi / phi
        
        for i in range(N):
            # Calculate spherical coordinates using the Fibonacci grid formula
            theta = np.arccos(1 - 2 * (i + 0.5) / N)
            phi_i = (golden_angle * i) % (2 * np.pi)
            points[i] = [theta, phi_i]
        
        return points

    @staticmethod
    def get_patch_pixels(image, side_length):
        """
        Extracts a square patch of pixels from the center of an image. The patch is 
        rotated by 45 degrees before extraction.
        """
        # Rotate the image by 45 degrees without changing the shape
        rotated_image = rotate(image, 45, reshape=False)

        # Determine the center of the rotated image
        x_center, y_center = rotated_image.shape[1] // 2, rotated_image.shape[0] // 2
        half_side = side_length // 2

        # Calculate the bounds of the patch
        x_start = max(x_center - half_side, 0)
        x_end = min(x_center + half_side, rotated_image.shape[1])
        y_start = max(y_center - half_side, 0)
        y_end = min(y_center + half_side, rotated_image.shape[0])

        # Extract the patch
        patch_pixels = rotated_image[y_start:y_end, x_start:x_end]
        
        return patch_pixels