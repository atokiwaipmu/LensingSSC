import numpy as np
import matplotlib.pyplot as plt

# Path to the PDF data file
data_file = '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/tiled/PDF/kappa_zs0.5_smoothed_s10_Counts_kappa_width0.1_200Kappabins.dat'

# Load the data
try:
    counts = np.loadtxt(data_file)
except Exception as e:
    print(f"Failed to load the data file: {e}")
    exit(1)

# Define the bins for the histogram (assuming the bins are from -0.1 to 0.1 with 200 bins)
bins = np.linspace(-0.1, 0.1, 200)

# Plot the PDF
plt.figure(figsize=(10, 6))
plt.plot(bins, counts, drawstyle='steps-post')
plt.xlabel('Kappa')
plt.ylabel('Probability Density')
plt.title('PDF of Kappa Map Smoothed s=10')
plt.grid(True)

# Save the figure
output_file = 'kappa_zs0.5_smoothed_s10_PDF.png'
plt.savefig(output_file, bbox_inches='tight')

# Show the plot
plt.show()
