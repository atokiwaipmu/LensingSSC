

import argparse
import yaml
import re
import glob
import os
import logging
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
import multiprocessing as mp
from tqdm import tqdm
import healpy as hp
from scipy.ndimage import rotate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class PatchProcessor:
    def __init__(self, npatch=273, patch_size=10, xsize=2048):
        logging.info(f"Initializing PatchProcessor with npatch={npatch}, patch_size={patch_size}")
        self.npatch = npatch
        self.patch_size = patch_size
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.reso = patch_size * 60 / xsize

    def make_patches(self, input_map):
        valid_points = self._get_valid_points()
        points_lonlatdeg = np.array([hp.rotator.vec2dir(hp.ang2vec(center[0], center[1]), lonlat=True) for center in valid_points])
        patches = [self._make_patch_worker(input_map, point) for point in points_lonlatdeg]
        return np.array(patches).astype(np.float32)

    def _make_patch_worker(self, input_map, point):
        patch = hp.gnomview(
            input_map,
            nest=False,
            rot=point,
            xsize=self.xsize * self.padding,
            reso=self.reso,
            return_projected_map=True,
            no_plot=True
        )
        return FibonacciHelper.get_patch_pixels(patch, self.xsize).astype(np.float32)
    
    def _get_valid_points(self):
        points = FibonacciHelper.fibonacci_grid_on_sphere(self.npatch)
        radius = np.radians(self.patch_size) * np.sqrt(2)
        valid_points = points[(points[:, 0] < np.pi - radius) & (points[:, 0] > radius)]
        return valid_points
    
class FlatPatchAnalyser:
    def __init__(self, pp: PatchProcessor, nbin=15, lmin=300, lmax=3000):
        logging.info(f"Initializing FlatPatchAnalyser with nbin={nbin}, lmin={lmin}, lmax={lmax}")
        self.pp = pp
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.binwidth = self.bins[1] - self.bins[0]

    def process_patches(self, patches_kappa, patches_snr, num_processes=mp.cpu_count()):
        with mp.Pool(processes=num_processes) as pool:
            datas = pool.starmap(self._process_patch, zip(patches_kappa, patches_snr))
        return np.array(datas).astype(np.float32)
    
    def _process_patch(self, patch_pixels, patch_snr_pixels):
        """
        Processes a single patch, computing various statistics (bispectrum, power spectrum, peak counts, etc.).
        """
        # Process kappa (convergence) map
        conv_map = ConvergenceMap(patch_pixels, angle=self.pp.patch_size * u.deg)
        squeezed_bispectrum = self._compute_bispectrum(conv_map)
        cl_power_spectrum = self._compute_power_spectrum(conv_map)
        
        # Process SNR map
        snr_map = ConvergenceMap(patch_snr_pixels, angle=self.pp.patch_size * u.deg)
        pdf_vals = self._compute_pdf(snr_map)
        peaks = self._compute_peak_statistics(snr_map, is_minima=False)
        minima = self._compute_peak_statistics(snr_map, is_minima=True)
        
        # Concatenate all computed statistics
        data_tmp = np.hstack([squeezed_bispectrum, cl_power_spectrum, pdf_vals, peaks, minima])
        return data_tmp

    def _compute_bispectrum(self, conv_map: ConvergenceMap):
        _, squeezed = conv_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')
        return squeezed
    
    def _compute_power_spectrum(self, conv_map: ConvergenceMap):
        _, cl = conv_map.powerSpectrum(self.l_edges)
        return cl
    
    def _compute_pdf(self, snr_map: ConvergenceMap):
        _, pdf_vals = snr_map.pdf(self.bins)
        return pdf_vals
    
    def _compute_peak_statistics(self, snr_map: ConvergenceMap, is_minima=False):
        if is_minima:
            # Invert the map for minima computation
            snr_map = ConvergenceMap(-snr_map.data, angle=self.pp.patch_size * u.deg)

        height, positions = snr_map.locatePeaks(self.bins)
        height, positions = self._exclude_edges(height, positions)
        peaks = np.histogram(height, bins=self.bins)[0]
        peaks = peaks / np.sum(peaks) / self.binwidth
        return peaks
    
    def _exclude_edges(self, heights, positions):
        """
        Excludes edge values from the peak or minima positions to avoid boundary issues.
        """
        # Scale positions to the patch size and apply boundary mask
        tmp_positions = positions.value * self.pp.xsize / self.pp.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.pp.xsize - 1) & \
               (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.pp.xsize - 1)
        return heights[mask], tmp_positions[mask].astype(int)
    
class FullSkyAnalyser:
    def __init__(self, nside=8192, nbin=15, lmin=300, lmax=3000):
        logging.info(f"Initializing FullSkyAnalyser with nside={nside}, nbin={nbin}, lmin={lmin}, lmax={lmax}")
        self.nside = nside
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.binwidth = self.bins[1] - self.bins[0]

        self.ef = ExtremaFinder(nside=self.nside)

    def process_map(self, snr_map, cl):
        cl = self._continuous_to_discrete(cl)

        pdf_vals = self._compute_histogram(data=snr_map)

        self.ef.initialize()
        _, peak_amp, _, minima_amp = self.ef.find_extrema(snr_map)
        peaks = self._compute_histogram(data=peak_amp)
        minima = self._compute_histogram(data=minima_amp)

        data_tmp = np.hstack([cl, pdf_vals, peaks, minima])
        return data_tmp
    
    def _compute_histogram(self, data):
        hist = self.parallel_histogram(data=data, bins=self.bins)
        return hist / np.sum(hist) / self.binwidth
    
    def _continuous_to_discrete(self, cl_cont):
        ell_cont = np.arange(2, self.lmax + 1)
        ell_idx = np.digitize(ell_cont, self.l_edges, right=True)
        cl_count = np.bincount(ell_idx, weights=cl_cont[1:-1])
        ell_bincount = np.bincount(ell_idx)
        cl_disc = (cl_count/ell_bincount)[1:]
        return cl_disc
    
    @staticmethod
    def parallel_histogram(data, bins:np.ndarray, num_processes=mp.cpu_count()):
        data_chunks = np.array_split(data, len(data) // 10000)
        with mp.Pool(processes=num_processes) as pool:
            hist_chunks = pool.starmap(FullSkyAnalyser.compute_histogram, [(chunk, bins) for chunk in data_chunks])
        final_hist = np.sum(hist_chunks, axis=0)
        return final_hist
    
    @staticmethod
    def compute_histogram(data_chunk, bins):
        hist, _ = np.histogram(data_chunk, bins=bins)
        return hist
    
class ExtremaFinder:
    def __init__(self, nside=8192):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)

    def initialize(self):
        if not hasattr(self, 'neighbours'):
            self.ipix = np.arange(self.npix)
            self.neighbours = hp.get_all_neighbours(self.nside, self.ipix)


    def find_extrema(self, kappa_map:np.ndarray, num_processes=mp.cpu_count()):
        neighbour_vals = kappa_map[self.neighbours.T]
        pixel_val = kappa_map[self.ipix]

        neighbour_chunks = np.array_split(neighbour_vals, len(self.ipix) // 10000)
        pixel_val_chunks = np.array_split(pixel_val, len(self.ipix) // 10000)

        with mp.Pool(processes=num_processes) as pool:
            results = [pool.apply_async(self._find_extrema_worker, args=(pixel_val_chunks[i], neighbour_chunks[i])) for i in range(len(pixel_val_chunks))]
            extrema_chunks = []
            for result in tqdm(results, desc="Searching for extrema...", total=len(results)):
                extrema_chunks.append(result.get())

        peaks_chunks, minima_chunks = zip(*extrema_chunks)
        peaks = np.concatenate(peaks_chunks)
        minima = np.concatenate(minima_chunks)

        peak_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[peaks], lonlat=False)).T
        peak_amp = pixel_val[peaks]

        minima_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[minima], lonlat=False)).T
        minima_amp = pixel_val[minima]
        
        return peak_pos, peak_amp, minima_pos, minima_amp

    @staticmethod
    def _find_extrema_worker(pixel_val: np.ndarray, neighbour_vals: np.ndarray):
        peaks = np.all(np.tile(pixel_val, (8, 1)).T > neighbour_vals, axis=-1)
        minima = np.all(np.tile(pixel_val, (8, 1)).T < neighbour_vals, axis=-1)
        return peaks, minima

class NoiseGenerator:
    def __init__(self, ngal=30, nside=8192, epsilon=0.3):
        logging.info(f"Initializing NoiseGenerator with ngal={ngal}, nside={nside}, epsilon={epsilon}")
        self.epsilon = epsilon
        self.ngal = ngal
        self.nside = nside
        self.pixarea = hp.nside2pixarea(nside, degrees=True) * 60 ** 2  # arcmin^2
        self.npix = hp.nside2npix(nside)

    def generate_noise(self, seed=0):
        logging.info(f"Generating noise with seed {seed}")
        np.random.seed(seed)
        sigma = self.epsilon / np.sqrt(self.ngal * self.pixarea)
        noise_map = np.random.normal(loc=0, scale=sigma, size=(self.npix,))
        return noise_map

class KappaSmoother:
    def __init__(self, scale_angle=2):
        self.scale_angle = scale_angle

    def smooth_kappa_map(self, kappa_map):
        #!!! setting OMP_NUM_THREADS does not work in Cluster
        logging.info(f"Smoothing kappa map with scale angle {self.scale_angle}")

        logging.info(f"Setting OMP_NUM_THREADS to {mp.cpu_count()}")
        self._set_omp_num_threads(mp.cpu_count())
        self._reimport_libraries()

        smoothed_map = hp.smoothing(kappa_map, sigma=self.scale_angle / 60 * np.pi / 180)

        self._set_omp_num_threads(1)
        self._reimport_libraries()
        
        return smoothed_map
    
    @staticmethod
    def _set_omp_num_threads(n):
        #!!! setting OMP_NUM_THREADS does not work in Cluster
        raise NotImplementedError("Setting OMP_NUM_THREADS does not work in Cluster")
        os.environ["OMP_NUM_THREADS"] = f"{n}"

    @staticmethod
    def _reimport_libraries():
        #!!! setting OMP_NUM_THREADS does not work in Cluster
        global hp, np
        import healpy as hp
        import numpy as np

class SuffixGenerator:
    def __init__(self, pp: PatchProcessor = PatchProcessor(), ng: NoiseGenerator = None, ks: KappaSmoother = KappaSmoother()):
        self.pp = pp 
        self.ng = ng 
        self.ks = ks 

        self.oa = self.pp.patch_size
        self.sl = self.ks.scale_angle

        self.noiseless = True if self.ng is None else False
        self.ngal = self.ng.ngal if not self.noiseless else None

    def generate_patch_suffix(self, kappa_path):
        seed = self.extract_seed_from_path(kappa_path)
        zs = self.extract_redshift_from_path(kappa_path)
        suffix = {
            'patch_snr': f"s{seed}_zs{zs}_oa{self.oa}_sl{self.sl}_noiseless",
            'patch_kappa': f"s{seed}_zs{zs}_oa{self.oa}_noiseless"
        }
        if not self.noiseless:
            for key in suffix.keys():
                suffix[key] = suffix[key].replace('noiseless', f"ngal{self.ngal}")
        return suffix
    
    def generate_fullsky_suffix(self, kappa_path):
        seed = self.extract_seed_from_path(kappa_path)
        zs = self.extract_redshift_from_path(kappa_path)
        suffix = {
            'fullsky': f"s{seed}_zs{zs}_sl{self.sl}_noiseless"
        }
        if not self.noiseless:
            suffix['fullsky'] = suffix['fullsky'].replace('noiseless', f"ngal{self.ngal}")
        return suffix
    
    def generate_cls_suffix(self, kappa_path):
        seed = self.extract_seed_from_path(kappa_path)
        zs = self.extract_redshift_from_path(kappa_path)
        suffix = f"zs{zs}_s{seed}_noiseless"
        if not self.noiseless:
            suffix = suffix.replace('noiseless', f"ngal{self.ngal}")
        return suffix
    
    @staticmethod
    def extract_seed_from_path(path):
        # Regular expression to find the pattern '_s{seed_number}_'
        match = re.search(r'_s(\d+)_', path)
        if match:
            return int(match.group(1))
        else:
            logging.error("Seed number not found in the given path.")
            return None
        
    @staticmethod
    def extract_redshift_from_path(path):
        # Regular expression to find the pattern '_zs{redshift}' e.g. '_zs2.0'
        match = re.search(r'_zs(\d+\.\d+)', path)
        if match:
            return float(match.group(1))
        else:
            logging.error("Redshift value not found in the given path.")
            return None
        
    @staticmethod
    def extract_type_from_path(path, bbox_size=[3750, 5000], sbox_size=[625]):
        # Regular expression to find the pattern '_size{box_size}_'
        match = re.search(r'_size(\d+)_', path)
        if match:
            box_size = int(match.group(1))
            if box_size in bbox_size:
                return 'bigbox'
            elif box_size in sbox_size:
                return 'tiled'
            else:
                logging.error("Box size is not valid.")
                return None
        else:
            logging.error("Box size not found in the given path.")
            return None

class KappaProcessor:
    def __init__(self, datadir,
                pp: PatchProcessor = PatchProcessor(), 
                fpa: FlatPatchAnalyser = FlatPatchAnalyser(pp=PatchProcessor()), 
                fsa: FullSkyAnalyser = FullSkyAnalyser(), 
                ng: NoiseGenerator = None, 
                ks: KappaSmoother = KappaSmoother(),
                sg: SuffixGenerator = SuffixGenerator(),
                overwrite=False):
        self.datadir = datadir
        self.overwrite = overwrite
        self.pp = pp 
        self.fpa = fpa 
        self.fsa = fsa 
        self.ng = ng 
        self.ks = ks 
        self.sg = sg
        self.noiseless = True if self.ng is None else False  

        self._check_ifkappa_dir_exists()     
        self._prepare_output_path()

    def _check_ifkappa_dir_exists(self):
        self.kappa_map_paths = sorted(glob.glob(os.path.join(self.datadir, 'kappa', '*.fits')))
        if not self.kappa_map_paths:
            logging.error(f"No kappa maps found in {os.path.join(self.datadir, 'kappa')}")
            raise FileNotFoundError(f"No kappa maps found in {os.path.join(self.datadir, 'kappa')}")
        logging.info(f"Found {len(self.kappa_map_paths)} kappa maps")

    def _prepare_output_path(self):
        self.kappa_dir = os.path.join(self.datadir, 'kappa')
        self.smoothed_dir = os.path.join(self.datadir, 'smoothed_maps')
        self.patch_dir = os.path.join(self.datadir, 'patch')
        self.flat_dir = os.path.join(self.datadir, 'flat')
        self.cls_dir = os.path.join(self.datadir, 'cls')
        self.full_dir = os.path.join(self.datadir, 'fullsky')

        os.makedirs(self.smoothed_dir, exist_ok=True)
        os.makedirs(self.patch_dir, exist_ok=True)
        os.makedirs(self.flat_dir, exist_ok=True)
        os.makedirs(self.cls_dir, exist_ok=True)
        os.makedirs(self.full_dir, exist_ok=True)
        logging.info(f"Output directories created")

    def process(self):
        for idx in range(len(self.kappa_map_paths)):
            logging.info(f"Processing {idx}/{len(self.kappa_map_paths)} kappa maps")
            self._process_data(idx)

    def _process_data(self, idx):
        self._reset_variables()

        self.kappa_path = self.kappa_map_paths[idx]
        self._generate_outputs()

        if not os.path.exists(self.outputs["smoothed"]):
            self._load_kappa()
            self.smoothed_map = self.ks.smooth_kappa_map(self.kappa)
            hp.write_map(self.outputs["smoothed"], self.smoothed_map)
        
        if os.path.exists(self.outputs["analysis_flat"]) and not self.overwrite:
            logging.info(f"File {self.outputs['analysis_flat']} already exists. Skipping.")
        else:
            self._process_patch()

        if os.path.exists(self.outputs["analysis_full"]) and not self.overwrite:
            logging.info(f"File {self.outputs['analysis_full']} already exists. Skipping.")
        else:
            self._process_fullsky()

    def _reset_variables(self):
        self.kappa = None
        self.smoothed_map = None
        self.outputs = None

    def _process_patch(self):
        logging.info(f"Processing patches")
        patches_kappa = self._patch_existance(is_kappa=True)
        patches_snr = self._patch_existance(is_kappa=False)
        data = self.fpa.process_patches(patches_kappa, patches_snr)
        np.save(self.outputs["analysis_flat"], data)

    def _patch_existance(self, is_kappa=True):
        flag = os.path.exists(self.outputs["patch_kappa"]) if is_kappa else os.path.exists(self.outputs["patch_snr"])
        if flag:
            logging.info(f"Loading {'kappa' if is_kappa else 'snr'} patches")
            patches = np.load(self.outputs["patch_kappa"]) if is_kappa else np.load(self.outputs["patch_snr"])
        else:
            logging.info(f"Generating {'kappa' if is_kappa else 'snr'} patches")
            if is_kappa:
                if self.kappa is None: self._load_kappa()
                patches = self.pp.make_patches(self.kappa)
            else:
                if self.smoothed_map is None: self._load_smoothed()
                patches = self.pp.make_patches(self.smoothed_map)
            np.save(self.outputs["patch_kappa"], patches) if is_kappa else np.save(self.outputs["patch_snr"], patches)
        return patches
             
    def _process_fullsky(self):
        logging.info(f"Processing full sky map")
        if os.path.exists(self.outputs["cls"]):
            logging.info(f"Loading cls from {os.path.basename(self.outputs['cls'])}")
            cl = hp.read_cl(self.outputs["cls"])
        else:
            cl = self._calculate_cls()
            hp.write_cl(self.outputs["cls"], cl)
            logging.info(f"Saved cls to {os.path.basename(self.outputs['cls'])}")

        if self.smoothed_map is None: self._load_smoothed()
        data = self.fsa.process_map(self.smoothed_map, cl)
        np.save(self.outputs["analysis_full"], data)

    def _calculate_cls(self):
        #!!! setting OMP_NUM_THREADS does not work in Cluster
        logging.info(f"Setting OMP_NUM_THREADS to {mp.cpu_count()}")
        KappaSmoother._set_omp_num_threads(mp.cpu_count())
        KappaSmoother._reimport_libraries()

        if self.kappa is None: self._load_kappa() 
        logging.info(f"Calculating cls")
        cl = hp.anafast(self.kappa, lmax=self.fsa.lmax)

        logging.info(f"Setting back OMP_NUM_THREADS to 1")
        KappaSmoother._set_omp_num_threads(1)
        KappaSmoother._reimport_libraries()
        return cl

    def _load_kappa(self):
        logging.info(f"Loading kappa map from {os.path.basename(self.kappa_path)}")
        self.kappa = hp.read_map(self.kappa_path)
        self.kappa = hp.reorder(self.kappa, n2r=True)
        if not self.noiseless:
            noise_map = self.ng.generate_noise(seed=SuffixGenerator.extract_seed_from_path(self.kappa_path))
            self.kappa = self.kappa + noise_map
            logging.info(f"Noise map added to kappa map")
    
    def _load_smoothed(self):
        logging.info(f"Loading smoothed map from {os.path.basename(self.outputs['smoothed'])}")
        self.smoothed_map = hp.read_map(self.outputs["smoothed"])
        self.smoothed_map = self.smoothed_map / np.std(self.smoothed_map)
        logging.info(f"Smoothed map loaded and normalized to unit variance")

    def _generate_outputs(self):
        suffix_fullsky = self.sg.generate_fullsky_suffix(self.kappa_path)
        suffix_patch = self.sg.generate_patch_suffix(self.kappa_path)
        suffix_cls = self.sg.generate_cls_suffix(self.kappa_path)

        self.outputs = {
            "analysis_flat": os.path.join(self.flat_dir, f'analysis_sqclpdpm_{suffix_patch["patch_snr"]}.npy'),
            "analysis_full": os.path.join(self.full_dir, f'fullsky_clpdpm_{suffix_fullsky["fullsky"]}.npy'),
            "patch_snr": os.path.join(self.patch_dir, f'snr_patches_{suffix_patch["patch_snr"]}.npy'),
            "patch_kappa": os.path.join(self.patch_dir, f'kappa_patches_{suffix_patch["patch_kappa"]}.npy'),
            "smoothed": os.path.join(self.smoothed_dir, f'kappa_smoothed_{suffix_fullsky["fullsky"]}.fits'),
            "cls": os.path.join(self.cls_dir, f'cl_{suffix_cls}.fits')
        }

def classes_from_config(config, noiseless=False):
    pp = PatchProcessor(**config.get('PatchProcessor', {}))
    analyser_common_config = config.get('AnalyserCommon', {})
    fpa = FlatPatchAnalyser(pp, **analyser_common_config)
    fsa = FullSkyAnalyser(nside=config['NoiseGenerator']['nside'], **analyser_common_config)
    ng = NoiseGenerator(**config.get('NoiseGenerator', {})) if not noiseless else None
    ks = KappaSmoother(**config.get('KappaSmoother', {}))
    sg = SuffixGenerator(pp=pp, ng=ng, ks=ks)
    return pp, fpa, fsa, ng, ks, sg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory of convergence maps')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--noiseless', action='store_true', help='Noiseless simulation')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pp, fpa, fsa, ng, ks, sg = classes_from_config(config, args.noiseless)

    kp = KappaProcessor(args.datadir, pp=pp, fpa=fpa, fsa=fsa, ng=ng, ks=ks, sg=sg, overwrite=args.overwrite)
    kp.process()