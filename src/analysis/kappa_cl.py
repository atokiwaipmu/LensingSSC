
import os
from datetime import date
from pylab import *
import numpy as np
import healpy as hp
import bigfile
import matplotlib.pyplot as plt
from .calc_cl import calc_cl
import argparse

### accuracy parameters
lmax=5000

def gen_comp(file, recalculate=False):
    """
    Generate and compare convergence power spectra from simulations and theory.

    Parameters:
        zs (float): Source redshift.
        file (str): Path to the simulation file.

    Returns:
        None
    """
    f = bigfile.File(file)

    nside = f['kappa'].attrs['nside'][0]
    zmin = f['kappa'].attrs['zlmin'][0]
    zmax = f['kappa'].attrs['zlmax'][0]
    zs = f['kappa'].attrs['zs'][0]

    print('nside =', nside)
    print('redshifts =', zs)

    lmax = min([1000, nside])
    ell_sim = np.arange(lmax + 1)

    fname = f'{file}/kappa_cl_z{zs:.2f}.npy'
    if not os.path.isfile(fname):
        kappa = hp.pixelfunc.reorder(f['kappa'][:], n2r=True)
        np.save(fname, kappa)

    fn_cl = f'{file}/kappa_cl_z{zs:.2f}.npz'
    if not os.path.isfile(fn_cl) or recalculate:
        if os.path.isfile(fname):
            kappa = np.load(fname)
        else:
            kappa = hp.pixelfunc.reorder(f['kappa'][:], n2r=True)
        cl = hp.anafast(kappa, lmax=lmax)
        np.savez(fn_cl, ell=ell_sim, cl=cl)

    # Load simulation cl
    data = np.load(fn_cl)
    ell_sim, cl_sim = data['ell'], data['cl']

    # Compute halofit curve
    fn_cl_camb = f'/lustre/work/akira.tokiwa/Projects/LensingSSC/results/halofit/kappa_cl_camb_z{zs:.2f}.npz'
    if not os.path.isfile(fn_cl_camb):
          ell, clkk, clkk_lin = calc_cl(zs)
          np.savez(fn_cl_camb, ell=ell, clkk=clkk, clkk_lin=clkk_lin)

    data = np.load(fn_cl_camb)
    ell, clkk, clkk_lin = data['ell'], data['clkk'], data['clkk_lin']

    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=16)
    ax.set_ylabel(r'$[\ell(\ell+1)/2\pi] C_\ell^\mathrm{kk}$', fontsize=16)

    ax.plot(ell, clkk_lin * ell * (ell + 1) / 2. / np.pi, 'c--', lw=3, alpha=0.5,
         label=f'Clkk linear (z_s={zs})')
    ax.plot(ell, clkk * ell * (ell + 1) / 2. / np.pi, 'g-', lw=3, alpha=0.5,
         label=f'Clkk halofit (z_s={zs})')
    ax.plot(ell_sim, cl_sim * ell_sim * (ell_sim + 1) / 2. / np.pi, 'k-', lw=1, alpha=0.6,
         label=f'Clkk CrownCanyon (z_s={zs})')

    ax.legend(loc=0, frameon=0)
    ax.set_title(str(date.today()))

    fig.savefig(f'{file}/kappa_cl_z{zs:.2f}.png', bbox_inches='tight')

if __name__ == '__main__':
    zs_list = [0.5, 1.0, 2.0, 3.0]
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    # choices=["tiled", "bigbox"] #['config_tiled_hp.json', 'config_bigbox_hp.json']
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    args = parser.parse_args()
    datadir = f'/lustre/work/akira.tokiwa/Projects/LensingSSC/results/{args.config}/wlen_hp'
    os.makedirs(datadir, exist_ok=True)
    for zs in zs_list:
        gen_comp(f"{datadir}/WL-{zs:.2f}-N8192", recalculate=False)