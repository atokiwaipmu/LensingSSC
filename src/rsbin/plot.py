import os
import sys
import numpy as np
import healpy as hp
import bigfile
import matplotlib.pyplot as plt


sys.path.append("/lustre/work/akira.tokiwa/Projects/LensingSSC/lib/code/CAMB")
import camb
from camb import model, initialpower
from datetime import date



class ConvergencePowerSpectrumAnalysis:
    def __init__(self, data_dir, redshift, cosmology_params):
        self.data_dir = data_dir
        self.redshift = redshift
        self.h, self.ombh2, self.omch2, self.A_s, self.n_s, self.lmax = cosmology_params

    def calc_cl(self, z_s):
        # Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        
        # Set cosmological parameters
        pars.set_cosmology(H0=100*self.h, ombh2=self.ombh2, omch2=self.omch2)
        pars.InitPower.set_params(As=self.A_s, ns=self.n_s)
        
        # Set the redshift and k_max for the power spectrum
        pars.set_matter_power(redshifts=[0.0], kmax=10.0)
        
        # Set the non-linear power spectrum to 'halofit'
        pars.NonLinear = model.NonLinear_both
        
        # Set the maximum multipole for the lensing spectrum
        pars.max_l = self.lmax
        
        # Set the redshift for the lensing source plane
        pars.SourceWindows = [camb.sources.GaussianSourceWindow(redshift=z_s)]
        
        # Calculate results for these parameters
        results = camb.get_results(pars)
        
        # Get the total and linear lensing potential power spectra
        ell = np.arange(0, self.lmax+1)
        clphiphi = results.get_lens_potential_cls(lmax=self.lmax, CMB_unit='muK')
        clkk = 1.0/4 * (ell+2.0)*(ell+1.0)*(ell)*(ell-1.0)*clphiphi[0]
        
        # Get the linear lensing potential power spectrum
        pars.NonLinear = model.NonLinear_none
        results_linear = camb.get_results(pars)
        clphiphi_lin = results_linear.get_lens_potential_cls(lmax=self.lmax, CMB_unit='muK')
        clkk_lin = 1.0/4 * (ell+2.0)*(ell+1.0)*(ell)*(ell-1.0)*clphiphi_lin[0]
        
        return ell[2:], clkk[2:], clkk_lin[2:]

    def analyze(self, output_dir):
        # Set the path to the data file
        file_path = os.path.join(self.data_dir, f"WL-{self.redshift:.2f}-N8192")
        
        # Open the bigfile
        f = bigfile.File(file_path)
        
        # Read the attributes
        nside = f['kappa'].attrs['nside'][0]
        zmin = f['kappa'].attrs['zlmin'][0]
        zmax = f['kappa'].attrs['zlmax'][0]
        zs = f['kappa'].attrs['zs'][0]
        
        print('nside = ', nside)
        print('redshifts = ', zs)
        
        lmax = min([1000, nside])
        ell_sim = np.arange(lmax + 1)
        
        fn_cl = os.path.join(self.data_dir, f"kappa_cl_z{zs:.2f}.npz")
        if not os.path.isfile(fn_cl):
            cl = hp.anafast(f['kappa'][:], lmax=lmax)
            np.savez(fn_cl, ell=ell_sim, cl=cl)
        
        # Load sim cl
        data = np.load(fn_cl)
        ell_sim, cl_sim = data['ell'], data['cl']
        
        # Compute halofit curve
        fn_cl_camb = os.path.join(self.data_dir, f"kappa_cl_camb_z{zs:.2f}.npz")
        if not os.path.isfile(fn_cl_camb):
            ell, clkk, clkk_lin = self.calc_cl(zs)
            np.savez(fn_cl_camb, ell=ell, clkk=clkk, clkk_lin=clkk_lin)
        
        data = np.load(fn_cl_camb)
        ell, clkk, clkk_lin = data['ell'], data['clkk'], data['clkk_lin']
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot the convergence power spectra
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$', fontsize=16)
        ax.set_ylabel(r'$[\ell(\ell+1)/2\pi] C_\ell^\mathrm{kk}$', fontsize=16)
        ax.plot(ell, clkk_lin * ell * (ell + 1) / 2. / np.pi, 'c--', lw=3, alpha=0.5, label='Clkk linear (z_s=%s)' % (zs))
        ax.plot(ell, clkk * ell * (ell + 1) / 2. / np.pi, 'g-', lw=3, alpha=0.5, label='Clkk halofit (z_s=%s)' % (zs))
        ax.plot(ell_sim, cl_sim * ell_sim * (ell_sim + 1) / 2. / np.pi, 'k-', lw=1, alpha=0.6, label='Clkk Simulation (z_s=%s)' % (zs))
        ax.legend(loc=0, frameon=0)
        ax.set_title(f"Convergence Power Spectrum (z = {zs:.2f}) - {date.today()}")
        ax.grid(True)
        fig.tight_layout()
        
        # Save the plot to the output directory
        output_file = os.path.join(output_dir, f"convergence_power_spectrum_z{zs:.2f}.png")
        fig.savefig(output_file)
        
        print(f"Convergence power spectrum plot saved to: {output_file}")

def main():
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
### accuracy parameters
lmax=5000

    cosmology_params = (h, ombh2, omch2, A_s, n_s, lmax)

    # Set the directory path and redshift
    data_directory = "/lustre/work/akira.tokiwa/globus/fastpm/rfof/rfof_proc4096_nc1024_size625_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8_LCDM_10tiled/wlen"
    redshift = 0.50

    # Set the output directory
    output_directory = "/lustre/work/akira.tokiwa/Projects/LensingSSC/img"

    # Create an instance of ConvergencePowerSpectrumAnalysis
    analysis = ConvergencePowerSpectrumAnalysis(data_directory, redshift, cosmology_params)

    # Run the analysis
    analysis.analyze(output_directory)

if __name__ == "__main__":
    main()