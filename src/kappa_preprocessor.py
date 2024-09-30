
import logging
import os
import numpy as np
import healpy as hp
from nbodykit.lab import BigFileCatalog
from astropy.cosmology import FlatLambdaCDM

class MassSheetProcessor:
    def __init__(self, datadir, extra_index=100, overwrite=False):
        """
        Initialize the mass sheet processor with cosmology settings and the mass sheets catalog.
        """
        self.datadir = datadir
        self.msheets = BigFileCatalog(os.path.join(datadir, "usmesh"), dataset="HEALPIX/")
        self.cosmo = FlatLambdaCDM(H0=0.6774 * 100, Om0=0.309)
        self.extra_index = extra_index
        self.overwrite = overwrite
        self._output_initialization()
        self._attr_initialization()

    def _output_initialization(self):
        self.outputdir = os.path.join(self.datadir, "mass_sheets")
        os.makedirs(self.outputdir, exist_ok=True)

    def _attr_initialization(self):
        self.aemitIndex_edges = self.msheets.attrs['aemitIndex.edges']
        self.aemitIndex_offset = self.msheets.attrs['aemitIndex.offset']
        self.a_interval = self.aemitIndex_edges[1] - self.aemitIndex_edges[0]
        self.npix = self.msheets.attrs['healpix.npix'][0]
        self.boxSize = self.msheets.attrs['BoxSize'][0]
        self.M_cdm = self.msheets.attrs['MassTable'][1]
        self.nc = self.msheets.attrs['NC'][0]
        self.rhobar = self.M_cdm * (self.nc / self.boxsize) ** 3 

    def preprocess(self, start=20, end=100):
        """
        Preprocess the mass sheets and save the processed data to the output directory.
        """
        prev_end = None
        for i in range(start, end):
            if self.aemitIndex_offset[i+1] == self.aemitIndex_offset[i+2]:
                logging.info(f"Sheet {i} is empty. Skipping...")
                continue

            save_path = os.path.join(self.outputdir, f"delta-sheet-{str(i).zfill(2)}.fits")
            if os.path.exists(save_path) and not self.overwrite:
                logging.info(f"Sheet {i} already exists. Skipping...")
                prev_end = None
                continue

            start, end = self.get_indices(i, start=prev_end)
            prev_end = end
            delta = self.get_mass_sheet(i, start, end)
            hp.write_map(save_path, delta.astype('float32'), nest=True)

    def get_indices(self, i, start=None):
        """
        Determine the start and end indices for processing the mass sheet.
        Adjust indices to handle boundaries and minimize errors.
        """
        if start is not None:
            end = self.aemitIndex_offset[i+2]
        else:
            start = self.aemitIndex_offset[i+1]
            end = self.aemitIndex_offset[i+2]
            if self.extra_index is not None:
                search_start_last = np.min([start + self.extra_index, end])
                aemit_start = self.msheets['Aemit'][start:search_start_last].compute()
                change_index_start = np.where(np.diff(aemit_start) == self.a_interval)[0]
                if len(change_index_start) != 0:
                    logging.info(f"Aemit {np.round(aemit_start[change_index_start[0]], 2):.2f} start changed from {start} to {start + change_index_start[0]}")
                    start += change_index_start[0]

        if start == end:
            return start, end
        
        logging.info(f"index start: {start}, end: {end}")

        if self.extra_index is not None:
            search_end_first = np.max([end - self.extra_index, start])
            aemit_end = self.msheets['Aemit'][search_end_first:end].compute()
            change_index_end = np.where(np.round(np.diff(aemit_end), 2) == self.a_interval)[0]      
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

        return delta