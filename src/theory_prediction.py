
import os
import yaml
import argparse
import numpy as np
import logging
from classy import Class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calc_cl(z_s, 
    ombh2 = 0.02247, 
    omch2 = 0.11923,
    h = 0.677,
    A_s = 2.10732e-9,
    n_s = 0.96824,
    lmax = 3000):
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

def main(savedir=None, lmax=3000, zs_list=[0.5, 1.0, 2.0, 3.0], overwrite=False):
    """
    Calculate the non-linear and linear Clkk values for the specified redshifts.

    Parameters:
        savedir (str): Directory to save the Clkk values.
        zs_list (list): List of redshift values.
    """
    if savedir is None:
        savedir = os.path.join(os.getcwd(), 'halofit')
    os.makedirs(savedir, exist_ok=True)

    for zs in zs_list:
        fn_out = os.path.join(savedir, f'kappa_zs{zs:.1f}_Clkk_ell_0_{lmax}.npz')
        if os.path.exists(fn_out) and not overwrite:
            logging.info(f"Clkk values for z_s={zs} already exist. Skipping...")
            continue

        ell, clkk, clkk_lin = calc_cl(zs)
        np.savez(fn_out, ell=ell, clkk=clkk, clkk_lin=clkk_lin)
        logging.info(f"Saved Clkk values to {fn_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument("--output", type=str, help="Output directory to save convergence maps")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--config', type=str, help='Configuration file path')

    args = parser.parse_args()

    # Initialize empty config
    config = {}
    # Load configuration from YAML if provided and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as file:
            try:
                config = yaml.safe_load(file)  # Load the configuration from YAML
            except yaml.YAMLError as exc:
                print("Warning: The config file is empty or invalid. Proceeding with default parameters.")
                print(exc)

    # Override YAML configuration with command-line arguments if provided
    config.update({
        'output': args.output if args.output else config.get('output', None),
        'overwrite': args.overwrite if args.overwrite else config.get('overwrite', False),
    })

    main(**config)