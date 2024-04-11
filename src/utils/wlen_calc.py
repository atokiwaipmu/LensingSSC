
import numpy

def inv_sigma(ds, dl, zl):
    ddls = 1 - numpy.multiply.outer(1 / ds, dl)
    ddls = ddls.clip(0)
    w = (100. / 3e5) ** 2 * (1 + zl)* dl
    inv_sigma_c = (ddls * w)
    return inv_sigma_c
    
def wlen(Om, dl, zl, ds, Nzs=1):
    """
        Parameters
        ----------
        dl, zl: distance and redshift of lensing objects
        
        ds: distance source plane bins. if a single scalar, do a delta function bin.
        
        Nzs : number of objects in each ds bin. len(ds) - 1 items
        
    """
    ds = numpy.atleast_1d(ds) # promote to 1d, sum will get rid of it
    integrand = 1.5 * Om * Nzs * inv_sigma(ds, dl, zl)
    Ntot = numpy.sum(Nzs)
    w_lensing = numpy.sum(integrand, axis=0) / Ntot
    
    return w_lensing