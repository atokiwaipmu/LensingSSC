import argparse
import logging
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from nbodykit.lab import BigFileCatalog

# Setup logging to provide information on the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CosmologySettings:
    def __init__(self, h=0.6774, om=0.309):
        """
        Initialize the cosmology with specified Hubble constant (h) and matter density (om).
        Default values correspond to the Planck15 cosmology.
        """
        self.h = h
        self.om = om
        self.cosmo = FlatLambdaCDM(H0=self.h * 100, Om0=self.om)

    def get_cosmology(self):
        """
        Return the cosmology object for further use in distance calculations.
        """
        return self.cosmo

class MassSheetProcessor:
    def __init__(self, msheets, cosmo, extra_index=100):
        """
        Initialize the mass sheet processor with cosmology settings and the mass sheets catalog.
        """
        self.msheets = msheets
        self.cosmo = cosmo
        self.extra_index = extra_index
        self.npix = msheets.attrs['healpix.npix'][0]  # Number of pixels in the HEALPix map
        self.boxsize = msheets.attrs['BoxSize'][0]  # Size of the simulation box
        self.M_cdm = msheets.attrs['MassTable'][1]  # Mass of dark matter particles
        self.nc = msheets.attrs['NC'][0]  # Number of cells in the grid
        self.rhobar = self.M_cdm * (self.nc / self.boxsize) ** 3  # Mean density

    def get_indices(self, i, start=None):
        """
        Determine the start and end indices for processing the mass sheet.
        Adjust indices to handle boundaries and minimize errors.
        """
        if start is not None:
            end = self.msheets.attrs['aemitIndex.offset'][i+2]
        else:
            start = self.msheets.attrs['aemitIndex.offset'][i+1]
            end = self.msheets.attrs['aemitIndex.offset'][i+2]
            if self.extra_index is not None:
                search_start_last = np.min([start + self.extra_index, end])
                aemit_start = self.msheets['Aemit'][start:search_start_last].compute()
                change_index_start = np.where(np.diff(aemit_start) == 0.01)[0]
                if len(change_index_start) != 0:
                    logging.info(f"Aemit {np.round(aemit_start[change_index_start[0]], 2):.2f} start changed from {start} to {start + change_index_start[0]}")
                    start += change_index_start[0]

        if start == end:
            return start, end
        
        logging.info(f"index start: {start}, end: {end}")

        if self.extra_index is not None:
            search_end_first = np.max([end - self.extra_index, start])
            aemit_end = self.msheets['Aemit'][search_end_first:end].compute()
            change_index_end = np.where(np.round(np.diff(aemit_end), 2) == 0.01)[0]      
            if len(change_index_end) != 0:
                logging.info(f"Aemit {np.round(aemit_end[change_index_end[0]], 2):.2f} end changed from {end} to {end - change_index_end[0]}")
                end -= change_index_end[0]

        return start, end

    def get_mass_sheet(self, i, start, end):
        """
        Process the mass sheet data for a given index 'i'.
        Calculate and return the density contrast, chi1, and chi2 distances.
        """
        logging.info(f"Processing sheet {i}: reading ID and Mass")

        pid = self.msheets['ID'][start:end].compute() 
        mass = self.msheets['Mass'][start:end].compute()
        ipix = pid % self.npix
        map_for_slice = np.bincount(ipix, weights=mass, minlength=self.npix)    

        a1, a2 = self.msheets.attrs['aemitIndex.edges'][i:i+2]
        z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
        chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
        volume_diff = 4 * np.pi * (chi1**3 - chi2**3) / (3 * self.npix)
        delta = map_for_slice / volume_diff / self.rhobar - 1

        return delta, chi1 / self.cosmo.h, chi2 / self.cosmo.h

def save_mass_sheets(msheets, datadir, overwrite=False):
    """
    Save the processed mass sheets to the specified directory.
    """
    processor = MassSheetProcessor(msheets, cosmo)
    os.makedirs(os.path.join(datadir, "mass_sheets"), exist_ok=True)
    prev_end = None

    for i in range(20, 100):
        if msheets.attrs['aemitIndex.offset'][i+1] == msheets.attrs['aemitIndex.offset'][i+2]:
            logging.info(f"Sheet {i} is empty. Skipping...")
            continue

        save_path = f"{datadir}/mass_sheets/delta-sheet-{i}.npz"
        if os.path.exists(save_path) and not overwrite:
            logging.info(f"Sheet {i} already exists. Skipping...")
            prev_end = None
            continue

        start, end = processor.get_indices(i, start=prev_end)
        prev_end = end
        delta, chi1, chi2 = processor.get_mass_sheet(i, start, end)
        np.savez(save_path, delta=np.array(delta, dtype="float32"), chi1=np.asarray([chi1]), chi2=np.asarray([chi2]))

def main(datadir, overwrite=False):
    """
    Main function to run the mass sheet processing and saving.
    """
    cat = BigFileCatalog(os.path.join(datadir, "usmesh/"), dataset="HEALPIX/")
    save_mass_sheets(cat, datadir, overwrite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory containing input files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mass sheets if they exist')
    args = parser.parse_args()

    cosmo = CosmologySettings().get_cosmology()  # Set the cosmology parameters
    main(args.datadir, args.overwrite)  # Run the main process
