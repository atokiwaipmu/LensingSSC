# LensingSSC

This repository contains the code used to produce the results in the paper "Lensing Super Sample Covariance" by Akira Tokiwa, Adrian E. Bayer, Jia Liu and Masahiro Takada.

## Goal of this research
- is our nominal setup of 1Gpc box sufficient in capturing SSC for HOS
- any HOS that are less sensitive to SSC?
- the redshift dependence of SSC contribution

### Setup
Follows the current plan for [LSST Y1 sims](https://docs.google.com/document/d/1wUc5joml9FFDijslaQth3-gvAAmvqU2NrVMbUUjOCyI/edit)
[Config](https://github.com/atokiwaipmu/LensingSSC/assets/106581872/df9ae051-ba15-4d29-a119-39e6c01aa430)

### Methods
By comparing between HOS (peaks, minima, PDFs, etc) measured from:
- (1) lensing maps generated using the large boxes (5Gpc), which include all modes and hence capture SSC
- (2) lensing maps generated using tiled small boxes (500Mpc), which miss modes bigger than 500Mpc (SSC contribution would be redshift dependent, where at lower z, you receive less contribution and at high z, you miss a lot more SSC)
Both (1) and (2) capture the usual non-Gaussian information (they would have the same resolution), only differ in large-scale modes. 

### Money plots
- ratio of (co-)variance between (1) and (2) 
- the ratio as a function of source redshift zs=[0.5, 1, 1.5, 2, 2.5, 3]
- the ratio as a function of small box size=[0.5, 1, 2, 3..] Gpc
- the ratio for all the relevant statistics

## URLs for the contents

- [Google Slide](https://docs.google.com/presentation/d/1pFHoPImFvwqnGVIz9azB3b0borwAqdktTghcXhbc2AA/edit#slide=id.g2943db99aa6_0_0)
- [Overleaf](https://www.overleaf.com/7662955643hwqybxnpdfmk#0a185c)

## Data
On NERSC HPSS
- Big box: `/home/a/abayer/rfof_proc262144_nc8192_size5000_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8`
- Heavily tiled box: `/home/a/abayer/rfof_proc4096_nc1024_size625_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8_LCDM_10tiled`

## References

### Codes
- Generate with born approximation: Use [wlen.py](https://github.com/bccp/simplehod/blob/master/scripts/wlen.py)
- Sample [notebook](https://github.com/liuxx479/CorrelatedSims/blob/master/hack_crowncanyon_kappa.ipynb) to test the resulting map
- Sample [job submission script](https://github.com/liuxx479/CorrelatedSims/blob/master/sbatch_gen.py ) (this is a code to generate submission script..)
