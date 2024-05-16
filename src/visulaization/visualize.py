
import json
from glob import glob
import os

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

def plot_extrema_on_patch(self, Nmap1, center, size_deg=10):
    """
    Plot a 10°x10° flat sky patch with circles indicating peaks and minima.
    
    Parameters
    ----------
    Nmap1 : int
        Index of the map to be used.
    center : tuple
        Center of the patch in (RA, DEC) in degrees.
    size_deg : float
        Size of the patch in degrees.
    """
    fn_header = os.path.basename(self.filenames[Nmap1]).split('.')[0]
    dir_peaks = os.path.join(self.dir_results, 'peaks')
    dir_minima = os.path.join(self.dir_results, 'minima')
    
    fn_peaks = os.path.join(dir_peaks, f'{fn_header}_Nmap{Nmap1 + 1}_peaks_posRADEC_amp.dat')
    fn_minima = os.path.join(dir_minima, f'{fn_header}_Nmap{Nmap1 + 1}_minima_posRADEC_amp.dat')
    
    peaks = np.loadtxt(fn_peaks)
    minima = np.loadtxt(fn_minima)

    ra_center, dec_center = center
    
    # Project HEALPix map to a flat sky map
    m = self.mapbins[Nmap1]
    hp_proj = hp.projector.GnomonicProj(rot=center, xsize=size_deg*60, reso=1)
    img = hp_proj.projmap(m, lambda x, y, z: hp.vec2pix(self.nside, x, y, z))

    # Convert RA, DEC to plot coordinates
    peak_x, peak_y = hp_proj.ang2xy(peaks[:, 0], peaks[:, 1], lonlat=True)
    minima_x, minima_y = hp_proj.ang2xy(minima[:, 0], minima[:, 1])

    # Plot the patch and overlay circles for peaks and minima
    plt.figure(figsize=(10, 10))
    plt.imshow(img, origin='lower', extent=[-size_deg/2, size_deg/2, -size_deg/2, size_deg/2], cmap='viridis')
    plt.colorbar(label='Kappa')

    plt.scatter(peak_x, peak_y, marker='o', edgecolor='red', facecolor='none', s=100, label='Peaks')
    #plt.scatter(minima_x, minima_y, marker='o', edgecolor='blue', facecolor='none', s=100, label='Minima')

    plt.xlim(-size_deg/2, size_deg/2)
    plt.ylim(-size_deg/2, size_deg/2)

    plt.xlabel('RA (degrees)')
    plt.ylabel('DEC (degrees)')
    plt.legend()
    plt.title(f'10°x10° Patch Centered at RA={ra_center}°, DEC={dec_center}°')
    plt.savefig('/lustre/work/akira.tokiwa/Projects/LensingSSC/img/patch.png')
    plt.show()

def plot_power_spectra(data_dir, halofit_dir, img_dir, zs):
    """
    Generate and save a plot of power spectra.

    Parameters:
        file_path (str): Path to the data directory.
        zs (float): Source redshift.

    Returns:
        None
    """
    ell_sim, cl_sim = compute_and_save_cl(data_dir, zs)
    ell_int_sim, cl_int_sim = compute_and_save_cl(data_dir, zs, intg=True)
    ell, clkk, clkk_lin = compute_and_save_cambcl(halofit_dir, zs)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=16)
    ax.set_ylabel(r'$[\ell(\ell+1)/2\pi] C_\ell^\mathrm{kk}$', fontsize=16)

    ax.plot(ell, clkk_lin * ell * (ell + 1) / 2. / np.pi, 'c--', lw=3, alpha=0.5, label=f'Clkk linear (z_s={zs})')
    ax.plot(ell, clkk * ell * (ell + 1) / 2. / np.pi, 'g-', lw=3, alpha=0.5, label=f'Clkk halofit (z_s={zs})')
    ax.plot(ell_sim, cl_sim * ell_sim * (ell_sim + 1) / 2. / np.pi, 'k-', lw=1, alpha=0.6, label=f'Clkk CrownCanyon (z_s={zs})')
    ax.plot(ell_int_sim, cl_int_sim * ell_int_sim * (ell_int_sim + 1) / 2. / np.pi, 'r-', lw=1, alpha=0.6, label=f'Clkk CrownCanyon int (z_s={zs})')

    ax.legend(frameon=False)
    ax.set_title(str(date.today()))
    logging.info(f"Saving plot to {img_dir}/kappa_cl_z{zs:.1f}.png")
    fig.savefig(f'{img_dir}/kappa_cl_z{zs:.1f}.png', bbox_inches='tight')

def main():
    config_file = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    path = config["destination"]
    save_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/img"

    files = glob(path + "/*.npz")

    for file in files:
        data = np.load(file)
        kappa = data["kappa"]
        Nm = data["Nm"]
        nsources = kappa.shape[0]
        zs = float(file.split("_")[-1].split(".")[0])

        
        for i in range(nsources):
            fig = plt.figure(figsize=(10, 5))
            hp.orthview(kappa[i], title="kappa %02.2f" % zs, sub=(1, 2, 1), fig=fig)
            hp.orthview(Nm[i], title="Nm %02.2f" % zs, sub=(1, 2, 2), fig=fig)
            fig.savefig(save_dir + "/kappa_%02.2f_%02.2f.png" % (zs, i))
            plt.close()

if __name__ == "__main__":
    main()