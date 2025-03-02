import numpy as np
from typing import Tuple, Callable
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const, units as u
from astropy.cosmology import z_at_value


def compute_weight_function(chi: float,
                            cosmo: FlatLambdaCDM,
                            zs: float) -> float:
    """
    Compute the weight function for weak lensing convergence.

    Parameters
    ----------
    chi : float
        Comoving distance for the lens plane in Mpc.
    cosmo : FlatLambdaCDM
        Cosmology object.
    zs : float
        Source redshift.

    Returns
    -------
    float
        Weight function value at comoving distance chi.
    """
    chis = cosmo.comoving_distance(zs).value  # in Mpc
    # Convert H0 from km/s/Mpc to 1/Mpc
    H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)
    z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    dchi = np.clip(1.0 - chi / chis, 0.0, None)
    return 1.5 * cosmo.Om0 * (H0 ** 2) * (1.0 + z) * chi * dchi


def compute_wlen_integral(chi1: float,
                          chi2: float,
                          wlen_func: Callable[[float, FlatLambdaCDM, float], float],
                          cosmo: FlatLambdaCDM,
                          zs: float) -> float:
    """
    Compute the weak lensing integral for a single mass sheet.

    Parameters
    ----------
    chi1 : float
        Lower comoving distance bound in Mpc.
    chi2 : float
        Upper comoving distance bound in Mpc.
    wlen_func : callable
        Weight function (e.g., compute_weight_function).
    cosmo : FlatLambdaCDM
        Cosmology object.
    zs : float
        Source redshift.

    Returns
    -------
    float
        The integrated weak lensing contribution from chi1 to chi2.
    """
    # Use an approximate midpoint for the integral
    chi_mid = 0.75 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)
    dchi = chi1 - chi2
    return wlen_func(chi_mid, cosmo, zs) * dchi


def index_to_chi_pair(index: int, cosmo: FlatLambdaCDM) -> Tuple[float, float]:
    """
    Convert a mass sheet index to a pair of comoving distance bounds.

    Parameters
    ----------
    index : int
        Index representing the mass sheet.
    cosmo : FlatLambdaCDM
        Cosmology object.

    Returns
    -------
    Tuple[float, float]
        Lower and upper comoving distances (chi1, chi2) in Mpc.
    """
    a1, a2 = 0.01 * index, 0.01 * (index + 1)
    z1, z2 = (1.0 / a1 - 1.0), (1.0 / a2 - 1.0)
    chi1, chi2 = cosmo.comoving_distance([z1, z2]).value  # in Mpc
    return chi1, chi2