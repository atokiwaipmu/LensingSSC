
import numpy as np
import healpy as hp
from scipy.ndimage import rotate
import math

def fibonacci_grid_on_sphere(N):
    """
    Generate an array of points on a Fibonacci grid on a sphere.

    Parameters:
    N (int): Number of points to distribute on the sphere.

    Returns:
    np.ndarray: Array of shape (N, 2) where each row contains (theta, phi) coordinates of a point.
    """
    points = np.zeros((N, 2))
    
    phi = (np.sqrt(5) + 1) / 2  # Golden ratio
    golden_angle = 2 * np.pi / phi
    
    for i in range(N):
        theta = np.arccos(1 - 2 * (i + 0.5) / N)
        phi = (golden_angle * i) % (2 * np.pi)
        points[i] = [theta, phi]
    
    return points



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

class opt_patchnum():
    def __init__(self, nside=1024):
        self.nside = nside


    def optimal_N(self, patch_size, N=280):
        flag = True
        radius = np.radians(patch_size) * np.sqrt(2)
        while flag:
            print("testing N = ", N)
            points = fibonacci_grid_on_sphere(N)
            counts = np.zeros(hp.nside2npix(self.nside))
            for center in points[(points[:, 0] < np.pi - radius) & (points[:, 0] > radius)]:
                vertices = self.rotated_vertices(center, patch_size)
                vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
                ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
                counts[ipix] += 1
                if np.any(counts[ipix] > 1):
                    N -= 1
                    break
            if np.all(counts[ipix] <= 1):
                flag = False
        return N

    def vertices_from_center(center, patch_size):
        """
        Generate the vertices of a patch centered at a given point on the sphere.
        
        Parameters:
        center (np.ndarray): Array of shape (2,) containing the (theta, phi) coordinates of the center of the patch.
        patch_size (float): Size of the patch in degrees.
        
        Returns:
        np.ndarray: Array of shape (4, 2) containing the (theta, phi) coordinates of the vertices of the patch.
        """
        theta, phi = center
        half_size = np.radians(patch_size) / np.sqrt(2)
        
        vertices = np.array([
            [theta, phi + half_size * np.sin(theta)],
            [theta + half_size, phi],
            [theta, phi - half_size  * np.sin(theta)],
            [theta - half_size, phi]
        ])
        
        return vertices

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

    def spherical_to_cartesian(theta, phi, r=1):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def rotation_matrix(self, theta, phi):
        """
        Generate a rotation matrix for rotating a point on the sphere by theta and phi.
        """
        R_phi = self.M([0, 0, 1], phi)
        R_theta = self.M(np.array([np.sin(phi), -np.cos(phi), 0]), np.pi/2 - theta)
        return np.dot(R_theta, R_phi)

    def rotate_point(self, point, theta, phi):
        """
        Rotate a point on the sphere by theta and phi.
        """
        x, y, z = self.spherical_to_cartesian(*point)
        x_rot, y_rot, z_rot = np.dot(self.rotation_matrix(theta, phi), np.array([x, y, z]))
        return self.cartesian_to_spherical(x_rot, y_rot, z_rot)[1:]

    def rotated_vertices(self, center, patch_size):
        """
        Generate the vertices of a patch centered at a given point on the sphere, rotated by theta and phi.
        """
        theta, phi = center
        vertices = self.vertices_from_center([np.pi/2, 0], patch_size)
        rotated_vertices = np.array([self.rotate_point(vertex, theta, phi) for vertex in vertices])
        return rotated_vertices


class patch_handler():
    def __init__(self, img, patch_size = 10, xsize = 1024, N=273, nest=True):
        self.img = img
        self.nest = nest

        self.patch_size = patch_size
        self.xsize = xsize
        self.reso = patch_size*60/xsize
        self.padding = 0.1 + np.sqrt(2)

        self.N = N
        self.points = fibonacci_grid_on_sphere(N)
        self.points_lonlatdeg = np.array([hp.rotator.vec2dir(hp.ang2vec(center[0], center[1]), lonlat=True) for center in self.points])

    def perform_patch(self):
        patchs = []
        for point in self.points_lonlatdeg:
            tmp_patch = hp.gnomview(self.img, nest=self.nest, rot=point, xsize=self.xsize*self.padding, reso=self.reso,return_projected_map=True, no_plot=True)
            patch_pixels = get_patch_pixels(tmp_patch, self.xsize)
            patchs.append(patch_pixels)

        self.patchs = np.array(patchs)

    def get_patch(self, i):
        return self.patchs[i]

    


    
