-- parameter file
------ Size of the simulation -------- 

-- For Testing
nc = 256
boxsize = 625

-------- Time Sequence ----
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga
-- time_step = linspace(0.01, 1.0, 10)
n_steps = 10    --? first use 10 as a trial to test i/o
ai = 0.1        --?
af = 1
time_step = linspace(ai, af, n_steps)

output_redshifts= {} --hack {2.,1.5,1.,0.5,0.}  -- redshifts of output  --?

-- Cosmology (TNG) --
Omega_m = 0.3089
h       = 0.6774

-- Start with a linear density field
-- Power spectrum of the linear density field: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter
read_powerspectrum= 'z1_pk.dat'  --z1 means z=0
linear_density_redshift = 0 -- the redshift of the linear density field
random_seed = tonumber(args[1])
particle_fraction = 1.0   --? 0.05, sub-sampling

--fof
--increasing fof_linkinglength can solve the problem that there is no data in fof file. but increasing to much will lead to crush of simulation
fof_linkinglength = 0.2
fof_nmin = 22            --?
fof_kdtree_thresh = 8 --smaller uses more memory but fof runs faster

--rfof
rfof_kdtree_thresh = 8
rfof_linkinglength = 0.2
rfof_nmin = 22            --10 doesn't work
rfof_l1 = 0.25
rfof_l6 = 0.235
rfof_a1 = 0.012
rfof_a2 = 0.06
rfof_b1 = 4.28
rfof_b2 = 2.17

--lightcone
dh_factor = 1.0 -- <~ Scale Hubble distance to amplify the lightcone effect. cannot be too small either
lc_octants = {0, 1, 2, 3, 4, 5, 6, 7} --卦限
lc_fov = 360   -- full sky
--lc_glmatrix = fastpm.translation(-128, -128, -128)
--lc_amin = 0.28 -- a=0.2857142857142857 is z=2.5 --   -- was 0.2 z=4, min scale factor for truncation of lightcone
lc_amin = 0.28   -- z=1
--lc_amax = 0.995 -- max scale factor for truncation of lightcone
lc_amax = 1.0

lc_usmesh_tiles = fastpm.outerproduct({-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
-- lc_usmesh_tiles = fastpm.outerproduct({-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,})
lc_usmesh_healpix_nside = 4096
lc_usmesh_nslices = 100
lc_usmesh_fof_padding = 10.0   --? roughly the size of a halo.
lc_usmesh_alloc_factor = 32.   --? allocation factor for the unstructured mesh, relative to alloc_factor. increasing it?
lc_usmesh_ell_limit = 10000

--pgd
pgdc = false

--fnl
--f_nl_type = "local"
--scalar_amp = 2.130624e-9
--scalar_pivot = 0.05
--scalar_spectral_index = 0.9667
--f_nl = 80.0
--kmax_primordial_over_knyquist = 0.25


-------- Approximation Method ---------------
force_mode = "fastpm"
--kernel_type = "1_4"  --?  "3_4"
force_softening_type = "none"

remove_cosmic_variance = true   --CHANGE

growth_mode = "LCDM"            --CHANGE

pm_nc_factor = 2   --? A list of {a, PM resolution}, {{0.0, 1}, {0.001, 3}} 
lpt_nc_factor = 2  --? PM resolution use in lpt and linear density field

np_alloc_factor= 4      -- Amount of memory allocated for particle   --?

-------- Output ---------------

loc = "/scratch/09665/atokiwa/test"

filename = string.format("/proc%d_nc%d_size%d_nsteps%d_s%d_nside%d_10tiled", os.get_nprocs(), nc, boxsize, n_steps, random_seed,lc_usmesh_healpix_nside)  --add time_step to

-- filename = string.format("rfof_proc%d_nc%d_size%d_nsteps%dlin_ldr%d_rcv%s_fst%s_pnf%d_lnf%d_s%d_dhf%.4f_tiled%.2f_fll_elllim_%d_npix_%d_rfofkdt_%d_fnl_%d", os.get_nprocs(), nc, boxsize, n_steps, linear_density_redshift, remove_cosmic_variance, force_softening_type, pm_nc_factor, lpt_nc_factor, random_seed, dh_factor, fof_linkinglength, lc_usmesh_ell_limit,lc_usmesh_healpix_nside,rfof_kdtree_thresh,f_nl)  --add time_step to 

-- Dark matter particle outputs (all particles)
--write_runpb_snapshot= "rfof/tpm"
--write_snapshot = loc .. filename .. "/snapshot"
-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = loc .. filename .. "/powerspec"

-- write lightcone
lc_write_usmesh = loc .. filename .. "/usmesh"

-- when running with lightcone only use fof or rfof in one run (FIXME Yu)
--if args[2] == "fof" then
--    write_fof = loc .. filename .. "/fof"
--elseif args[2] == "rfof" then
--    write_rfof = loc .. filename .. "/rfof"
--end