import healpy as hp
import numpy as np

# Define parameters
size_deg = 10

# Function to project the map
def project_map(m, nside, center):
    hp_proj = hp.projector.GnomonicProj(rot=center, xsize=size_deg*60, reso=1)
    return hp_proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z))

# Example map (m) and nside
m = hp.read_map('/lustre/work/akira.tokiwa/Projects/LensingSSC/results/tiled/data/kappa_zs0.5_smoothed_s5.fits')  # Read your map
nside = hp.get_nside(m)  # Get nside from the map

# Create a grid of center points for the patches
lons = np.arange(-180, 180, size_deg)
lats = np.arange(-90, 90, size_deg)

# Calculate the total number of plots
num_lons = len(lons)
num_lats = len(lats)

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        center = (lon, lat)
        projected_map = project_map(m, nside, center)
        np.save(f"/lustre/work/akira.tokiwa/Projects/LensingSSC/results/tiled/flatsky/kappa_zs0.5_smoothed_s5_center_{lon}_{lat}.npy", projected_map)
        print(f"Saved {i*num_lons + j + 1}/{num_lons*num_lats} patches")