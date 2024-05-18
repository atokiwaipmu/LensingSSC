
import os
import numpy as np
import logging
from classy import Class

from ..masssheet.ConfigData import ConfigData, ConfigAnalysis, ConfigCosmo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calc_cl(z_s, 
    ombh2 = 0.02247, 
    omch2 = 0.11923,
    h = 0.677,
    A_s = 2.10732e-9,
    n_s = 0.96824,
    lmax = 5000):
    """
    Calculate the non-linear and linear Clkk values using the Class module.

    Parameters:
        z_s (float): Selection mean redshift.
        ombh2 (float): Omega_b h^2 value.
        omch2 (float): Omega_cdm h^2 value.
        h (float): Hubble parameter.
        A_s (float): Primordial amplitude of scalar perturbations.
        n_s (float): Spectral index of primordial scalar perturbations.
        lmax (int): Maximum multipole moment.

    Returns:
        tuple: A tuple containing the following:
            - ell (array): Array of multipole moments.
            - clkk (array): Array of non-linear Clkk values.
            - clkk_lin (array): Array of linear Clkk values.
    """
    LambdaCDM = Class()
    LambdaCDM.set({'omega_b':ombh2,'omega_cdm':omch2,'h':h,'A_s':A_s,'n_s':n_s})
    LambdaCDM.set({'output':'mPk,sCl',
                   #'lensing':'yes',
                   'lensing':'no',
                   'P_k_max_1/Mpc':10.0,
                   'z_pk':0,
                   'reio_parametrization':'reio_none',
                   'l_switch_limber':100,
                   'selection':'dirac',
                   'selection_mean':z_s,
                   'l_max_lss':lmax,
                   #'perturb_sampling_stepsize':0.01,
                   'non linear':'halofit'
                  })

    LambdaCDM_linear = Class()
    LambdaCDM_linear.set({'omega_b':ombh2,'omega_cdm':omch2,'h':h,'A_s':A_s,'n_s':n_s})
    LambdaCDM_linear.set({'output':'mPk,sCl',
                   #'lensing':'yes',
                   'lensing':'no',
                   'P_k_max_1/Mpc':10.0,
                   'z_pk':0,
                   'reio_parametrization':'reio_none',
                   'l_switch_limber':100,
                   'selection':'dirac',
                   'selection_mean':z_s,
                   'l_max_lss':lmax,
                   #'perturb_sampling_stepsize':0.01
                  })

    # Run the calculations
    LambdaCDM.compute()
    LambdaCDM_linear.compute()

    # Get non-linear Clkk values
    cls2 = LambdaCDM.density_cl(lmax)
    ell = cls2['ell'][2:]
    clphiphi = cls2['ll'][0][2:]
    clkk = 1.0 / 4 * (ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0) * clphiphi

    # Get linear Clkk values
    cls2_lin = LambdaCDM_linear.density_cl(lmax)
    ell = cls2_lin['ell'][2:]
    clphiphi_lin = cls2_lin['ll'][0][2:]
    clkk_lin = 1.0 / 4 * (ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0) * clphiphi_lin

    return ell, clkk, clkk_lin

if __name__ == '__main__':
    config_data_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_data_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    config_cosmo_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_cosmo.json')
    config_cosmo = ConfigCosmo.from_json(config_cosmo_file)

    halofit_dir = os.path.join(config_analysis.resultsdir, 'halofit')
    os.makedirs(halofit_dir, exist_ok=True)

    for zs in config_data.zs_list:
        fn_out = os.path.join(halofit_dir , f'kappa_zs{zs:.1f}_Clkk_ell_0_{config_analysis.lmax}.npz')
        ell, clkk, clkk_lin = calc_cl(zs, ombh2=config_cosmo.ombh2, omch2=config_cosmo.omch2, h=config_cosmo.h, A_s=config_cosmo.A_s, n_s=config_cosmo.n_s, lmax=config_analysis.lmax)
        np.savez(fn_out, ell=ell, clkk=clkk, clkk_lin=clkk_lin)
        logging.info(f"Saved Clkk values to {fn_out}")