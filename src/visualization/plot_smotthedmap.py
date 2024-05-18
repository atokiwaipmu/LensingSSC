import healpy as hp
import matplotlib.pyplot as plt
import os

# File paths to the kappa maps
file_paths = [
    '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/bigbox/data/kappa_zs0.5_smoothed_s5.fits',
    '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/bigbox/data/kappa_zs0.5_smoothed_s8.fits',
    '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/bigbox/data/kappa_zs0.5_smoothed_s10.fits',
    '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/bigbox/data/kappa_zs0.5.fits'
]

# Titles for the plots
titles = [
    'Kappa Map Smoothed s=5',
    'Kappa Map Smoothed s=8',
    'Kappa Map Smoothed s=10',
    'Kappa Map'
]

# Directory to save the images
output_dir = '/lustre/work/akira.tokiwa/Projects/LensingSSC/img/bigbox'
os.makedirs(output_dir, exist_ok=True)

# Read, display, and save the maps
for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    try:
        print(f"Processing {file_path}")
        kappa_map = hp.read_map(file_path)
        
        # Create a figure for the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the kappa map using mollweide projection
        if i ==3:
            hp.mollview(kappa_map, title=title, cmap='jet', min=-0.006, max=0.006, nest=True)
        else:
            hp.mollview(kappa_map, title=title, cmap='jet', min=-0.006, max=0.006)
        
        # Add a color bar
        hp.graticule()
        
        # Save the figure
        output_file = os.path.join(output_dir, title.replace(' ', '_') + '.png')
        plt.savefig(output_file, bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
