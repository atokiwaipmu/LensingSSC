from classy import Class

######## check TT, Pk (z=0), Clkk (z=1) from class vs camb
ombh2 = 0.02247
omch2 = 0.11923

# LCDM parameters
A_s = 2.10732e-9
h=0.677
OmegaB = ombh2/h**2#0.046
OmegaM = omch2/h**2#0.309167
n_s = 0.96824
tau = 0.054 ## only for primary CMB, not used for now, for simplicity

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