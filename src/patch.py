
from ast import arg
import logging
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
import multiprocessing as mp
from scipy.special import erf
from pathlib import Path
import h5py

class PatchAnalyser:
    def __init__(self, 
                 ngal_list=[0, 7, 15, 30, 50], 
                 sl_list = [2, 5, 8, 10], 
                 nbin=8, 
                 lmin=300, 
                 lmax=3000, 
                 epsilon = 0.26, 
                 patch_size_deg=10, 
                 xsize=2048):
        
        self.nbin = nbin
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.nu = (self.bins[1:] + self.bins[:-1]) / 2

        self.lmin, self.lmax = lmin, lmax
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.binwidth = self.bins[1] - self.bins[0]

        self.ngal_list = ngal_list
        self.sl_list = sl_list
        self.epsilon = epsilon

        self.patch_size = patch_size_deg
        self.xsize = xsize
        self.pixarea_arcmin2 = (patch_size_deg * 60 / xsize)**2

    def process_patches(self, patches_kappa, num_processes=mp.cpu_count()):
        results = {}
        for ngal in self.ngal_list:
            logging.info(f"Processing kappa for ngal={ngal}")

            results[ngal] = {}
            with mp.Pool(processes=num_processes) as pool:
                datas = pool.starmap(self._process_kappa, zip(patches_kappa, [ngal] * len(patches_kappa)))
            
            datas = np.array(datas).astype(np.float32)
        
            results[ngal]["equilateral"] = datas[:, 0, :].astype(np.float32)
            results[ngal]["isosceles"] = datas[:, 1, :].astype(np.float32)
            results[ngal]["squeezed"] = datas[:, 2, :].astype(np.float32)
            results[ngal]["cl"] = datas[:, 3, :].astype(np.float32)

            logging.info(f"kappa processed for ngal={ngal}")

            for sl in self.sl_list:
                logging.info(f"Processing snr for ngal={ngal}, sl={sl}")

                results[ngal][sl] = {}
                with mp.Pool(processes=num_processes) as pool:
                    datas = pool.starmap(self._process_snr, zip(patches_kappa, [ngal] * len(patches_kappa), [sl] * len(patches_kappa)))

                pdfs, peaks, minimas, v0s, v1s, v2s, s0s, s1s = zip(*datas)
                
                results[ngal][sl]["pdf"] = np.array(pdfs).astype(np.float32)
                results[ngal][sl]["peaks"] = np.array(peaks).astype(np.float32)
                results[ngal][sl]["minima"] = np.array(minimas).astype(np.float32)
                results[ngal][sl]["v0"] = np.array(v0s).astype(np.float32)
                results[ngal][sl]["v1"] = np.array(v1s).astype(np.float32)
                results[ngal][sl]["v2"] = np.array(v2s).astype(np.float32)
                results[ngal][sl]["sigma0"] = np.array(s0s).astype(np.float32)
                results[ngal][sl]["sigma1"] = np.array(s1s).astype(np.float32)

                logging.info(f"snr processed for ngal={ngal}, sl={sl}")

        return results
    
    def _process_kappa(self, patch_pixels, ngal):
        if ngal == 0:
            pixels = patch_pixels
        else:
            noise_level = self.epsilon / np.sqrt(ngal * self.pixarea_arcmin2)
            noise_map = np.random.normal(0, noise_level, patch_pixels.shape)
            pixels = patch_pixels + noise_map

        conv_map = ConvergenceMap(pixels, angle=self.patch_size * u.deg)
        equilateral, isosceles, squeezed = self._compute_bispectrum(conv_map)
        cl = conv_map.powerSpectrum(self.l_edges)[1] * self.ell * (self.ell + 1) / (2 * np.pi)

        return [equilateral, isosceles, squeezed, cl]

    def _process_snr(self, patch_pixels, ngal, sl):
        if ngal == 0:
            pixels = patch_pixels
        else:
            noise_level = self.epsilon / np.sqrt(ngal * self.pixarea_arcmin2)
            noise_map = np.random.normal(0, noise_level, patch_pixels.shape)
            pixels = patch_pixels + noise_map

        conv_map = ConvergenceMap(pixels, angle=self.patch_size * u.deg)
        smoothed_map = conv_map.smooth(sl*u.arcmin).data
        sigma0 = np.std(smoothed_map)
        smoothed_map = ConvergenceMap(smoothed_map/sigma0, angle=self.patch_size * u.deg)
        pdf = smoothed_map.pdf(self.bins)[1]
        peaks = self._compute_peak_statistics(smoothed_map, is_minima=False)
        minima = self._compute_peak_statistics(smoothed_map, is_minima=True)
        v0, v1, v2, sigma1 = self._compute_minkowski_functionals(smoothed_map)

        return pdf, peaks, minima, v0, v1, v2, sigma0, sigma1

    def _compute_bispectrum(self, conv_map: ConvergenceMap):
        equilateral = conv_map.bispectrum(self.l_edges, configuration='equilateral')[1]
        isosceles = conv_map.bispectrum(self.l_edges, ratio=0.5, configuration='folded')[1]
        squeezed = conv_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')[1]

        equilateral = np.abs(PatchAnalyser._dimensionless_bispectrum(equilateral, self.ell))
        isosceles = np.abs(PatchAnalyser._dimensionless_bispectrum(isosceles, self.ell))
        squeezed = np.abs(PatchAnalyser._dimensionless_bispectrum(squeezed, self.ell))

        return equilateral, isosceles, squeezed
    
    def _compute_peak_statistics(self, snr_map: ConvergenceMap, is_minima=False):
        if is_minima:
            # Invert the map for minima computation
            snr_map = ConvergenceMap(-snr_map.data, angle=self.patch_size * u.deg)

        height, positions = snr_map.locatePeaks(self.bins)
        height, positions = self._exclude_edges(height, positions)
        peaks = np.histogram(height, bins=self.bins)[0]
        peaks = peaks / np.sum(peaks) / self.binwidth
        return peaks
    
    @staticmethod
    def compute_gradient(data):
        grad_y, grad_x = np.gradient(data)
        return [grad_y, grad_x]
    
    @staticmethod
    def compute_hessian(data):
        grad_y, grad_x = np.gradient(data)
        hess_yy, hess_yx = np.gradient(grad_y)
        hess_xy, hess_xx = np.gradient(grad_x)
        return [hess_xx, hess_yy, hess_xy]
    
    @staticmethod
    def Gaussian_MinkowskiFunctionals(nu, mu, sigma0, sigma1):
        V0 = 0.5 * (1.0 - erf((nu - mu) / (np.sqrt(2) * sigma0)))
        V1 = (sigma1 / (8.0 * sigma0 * np.sqrt(2))) * np.exp(-((nu - mu) ** 2) / (2.0 * sigma0 ** 2))
        V2 = ((nu - mu) * (sigma1 ** 2) / (sigma0 ** 3) /
            (2.0 * (2.0 * np.pi) ** 1.5)) * np.exp(-((nu - mu) ** 2) / (2.0 * sigma0 ** 2))
        return V0, V1, V2
    
    @staticmethod
    def compute_minkowski_functionals(data, nu, grad, hess):
        gradient_y, gradient_x = grad
        hessian_xx, hessian_yy, hessian_xy = hess

        denominator = gradient_x**2 + gradient_y**2
        s1 = np.sqrt(denominator)
        # Avoid division by zero
        denominator = np.where(denominator == 0, np.nan, denominator)
        frac = (2.0 * gradient_x * gradient_y * hessian_xy - 
                gradient_x**2 * hessian_yy - 
                gradient_y**2 * hessian_xx) / denominator
        frac = np.nan_to_num(frac)  # Replace NaNs with zero

        delta = np.diff(nu)

        # Initialize Minkowski functionals
        V0 = np.zeros(len(delta))
        V1 = np.zeros(len(delta))
        V2 = np.zeros(len(delta))

        # Normalize by total number of pixels
        total_pixels = data.size

        # Precompute all masks at once
        masks = [(data > lower) & (data < upper) for lower, upper in zip(nu[:-1], nu[1:])]
        masks = np.array(masks)

        V0 = np.array([np.sum(data > threshold) for threshold in nu[:-1]]) / total_pixels
        # Vectorized computation for V1 and V2
        for i in range(len(delta)):
            mask = masks[i]
            V1[i] = np.sum(s1[mask]) / (4.0 * delta[i] * total_pixels)
            V2[i] = np.sum(frac[mask]) / (2.0 * np.pi * delta[i] * total_pixels)

        return V0, V1, V2
    
    def _compute_minkowski_functionals(self, snr_map: ConvergenceMap):
        grad = self.compute_gradient(snr_map.data)
        hess = self.compute_hessian(snr_map.data)
        sigma1 = np.sqrt(np.mean(grad[0]**2 + grad[1]**2))

        v0, v1, v2 = self.compute_minkowski_functionals(snr_map.data, self.bins, grad, hess)

        return v0, v1, v2, sigma1
    
    def _exclude_edges(self, heights, positions):
        """
        Excludes edge values from the peak or minima positions to avoid boundary issues.
        """
        # Scale positions to the patch size and apply boundary mask
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize - 1) & \
               (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize - 1)
        return heights[mask], tmp_positions[mask].astype(int)
    
    @staticmethod
    def _dimensionless_bispectrum(bispec, ell):
        return bispec * ell**4 / (2*np.pi)**2
    
def save_results_to_hdf5(results, output_path):
    def _save_group(group, data):
        for key, value in data.items():
            # 数値キーを文字列に変換
            key_str = f"{key}"
            
            if isinstance(value, dict):
                # 辞書の場合は新しいグループを作成
                subgroup = group.create_group(key_str)
                _save_group(subgroup, value)
            else:
                # リストまたはnumpy配列の場合はデータセットとして保存
                if isinstance(value, list):
                    if len(value) > 0:
                        value = np.array(value)
                    else:
                        value = np.array([])
                try:
                    group.create_dataset(key_str, data=value, compression="gzip")
                except (TypeError, ValueError):
                    # 特殊なデータ型の場合は文字列として保存
                    group.create_dataset(key_str, data=str(value))

    with h5py.File(output_path, 'w') as f:
        _save_group(f, results)

def _save_results_to_hdf5(results, save_path):
    """
    Saves the results dictionary to an HDF5 file.

    Parameters:
    - results (dict): Nested dictionary containing the results.
    - save_path (Path): Path object indicating where to save the HDF5 file.
    """
    try:
        with h5py.File(save_path, 'w') as hf:
            for ngal, ngal_data in results.items():
                logging.info(f"Saving data for ngal={ngal}")
                ngal_group = hf.create_group(f'ngal_{ngal}')

                # Save 'cl', 'equilateral', 'isosceles', 'squeezed'
                for key in ['cl', 'equilateral', 'isosceles', 'squeezed']:
                    if key in ngal_data:
                        data = ngal_data[key]
                        if isinstance(data, np.ndarray):
                            # Ensure data is of a numeric type
                            if np.issubdtype(data.dtype, np.number):
                                dataset = ngal_group.create_dataset(
                                    key,
                                    data=data,
                                    compression="gzip",
                                    compression_opts=4
                                )
                                dataset.attrs['description'] = f'{key} data for ngal={ngal}'
                            else:
                                logging.error(f"Data type for {key} in ngal={ngal} is not numeric, but {data.dtype}. Skipping.")
                        else:
                            logging.error(f"Data for {key} in ngal={ngal} is not a NumPy array. Skipping.")

                # Iterate over 'sl' values
                for sl, sl_data in ngal_data.items():
                    if isinstance(sl, (int, float)):
                        logging.info(f"  Saving data for sl={sl}")
                        sl_group = ngal_group.create_group(f'sl_{sl}')

                        # Save 'pdf', 'peaks', 'minima', 'v0', 'v1', 'v2', 'sigma0', 'sigma1'
                        for sub_key in ['pdf', 'peaks', 'minima', 'v0', 'v1', 'v2']:
                            if sub_key in sl_data:
                                data = sl_data[sub_key]
                                if isinstance(data, np.ndarray):
                                    if np.issubdtype(data.dtype, np.number):
                                        sl_group.create_dataset(
                                            sub_key,
                                            data=data,
                                            compression="gzip",
                                            compression_opts=4
                                        )
                                        sl_group[sub_key].attrs['description'] = f'{sub_key} data for ngal={ngal}, sl={sl}'
                                    else:
                                        logging.error(f"Data type for {sub_key} in ngal={ngal}, sl={sl} is not numeric. Skipping.")
                                else:
                                    logging.error(f"Data for {sub_key} in ngal={ngal}, sl={sl} is not a NumPy array. Skipping.")

        logging.info(f"Successfully saved results to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {save_path}: {e}")
        raise

if __name__ == "__main__":
    import glob
    import sys
    import argparse

    argparser = argparse.ArgumentParser(description="Process convergence maps and compute statistics.")
    argparser.add_argument("--box_type", type=str, default="tiled", help="Type of box to process (tiled or bigbox)")
    argparser.add_argument("--zs", type=float, default=0.5, help="Source redshift to process")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dir = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/data/patches")
    output_base_dir = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/output")
    patch_analyser = PatchAnalyser()

    save_dir = output_base_dir / args.box_type / "stats"
    logging.info(f"Starting processing for box_type={args.box_type}, saving to {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
        
    # Use f-strings for consistent path formatting
    patches_kappa_paths = glob.glob(str(data_dir / args.box_type / f"zs{args.zs}/*.npy"))
    logging.info(f"Found {len(patches_kappa_paths)} files for zs={args.zs} in box_type={args.box_type}")
    
    if not patches_kappa_paths:
        logging.warning(f"No .npy files found for zs={args.zs} in {args.box_type}. Skipping.")

    for f in patches_kappa_paths:
        # Construct the save path
        filename = Path(f).name
        save_filename = filename.replace("patches", "stats").replace("_noiseless", "").replace(".npy", ".h5")
        save_path = save_dir / save_filename

        if save_path.exists() and not args.overwrite:
            logging.info(f"Skipping {Path(f).name} as {save_path} already exists.")
            continue

        logging.info(f"Processing {Path(f).name}")
        patches_kappa = np.load(f, mmap_mode='r')
        results = patch_analyser.process_patches(patches_kappa)

        logging.info(f"Saved results to {save_path}")
        save_results_to_hdf5(results, save_path)
        #sys.exit(0)  # Exit after processing one file for testing purposes