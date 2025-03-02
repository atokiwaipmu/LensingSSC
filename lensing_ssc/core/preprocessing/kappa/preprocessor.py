import logging
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import healpy as hp
from nbodykit.lab import BigFileCatalog
from astropy.cosmology import FlatLambdaCDM

from lensing_ssc.core.preprocessing.utils.indices_finder import IndicesFinder


class MassSheetProcessor:
    """Processes mass sheets for cosmological data visualization and analysis."""

    def __init__(self, datadir: Path, overwrite: bool = False) -> None:
        """
        Initialize the MassSheetProcessor.

        Args:
            datadir (Path): Directory containing the mass sheets data.
            overwrite (bool, optional): Overwrite existing processed files. Defaults to False.
        """
        self.datadir = datadir
        self.overwrite = overwrite

        self.msheets = BigFileCatalog(self.datadir / "usmesh", dataset="HEALPIX/")
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)

        self._initialize_output_directory()
        self._initialize_attributes()
        self.indices_df = self._load_precomputed_indices()

    def _initialize_output_directory(self) -> None:
        """Create the output directory for processed mass sheets."""
        self.output_dir = self.datadir / "mass_sheets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_attributes(self) -> None:
        """Initialize catalog attributes."""
        attrs = self.msheets.attrs
        self.aemit_index_edges = attrs['aemitIndex.edges']
        self.aemit_index_offset = attrs['aemitIndex.offset']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]
        self.npix = attrs['healpix.npix'][0]
        self.box_size = attrs['BoxSize'][0]
        self.m_cdm = attrs['MassTable'][1]
        self.nc = attrs['NC'][0]
        self.rhobar = self.m_cdm * (self.nc / self.box_size) ** 3
        self.seed = attrs['seed'][0]

    def _load_precomputed_indices(self) -> pd.DataFrame:
        """
        Load precomputed indices from CSV. If not found, run IndicesFinder.

        Returns:
            pd.DataFrame: DataFrame with columns 'sheet', 'start', and 'end'.
        """
        csv_path = self.datadir / f"preproc_s{self.seed}_indices.csv"
        if not csv_path.is_file():
            logging.error(f"Precomputed indices file not found: {csv_path}. Running IndicesFinder.")
            finder = IndicesFinder(self.datadir, self.seed)
            finder.find_indices()
        df = pd.read_csv(csv_path)
        if not {'sheet', 'start', 'end'}.issubset(df.columns):
            logging.error("CSV file is missing required columns: 'sheet', 'start', 'end'.")
            raise ValueError("CSV file must contain columns: 'sheet', 'start', 'end'")
        return df

    def preprocess(self) -> None:
        """
        Preprocess mass sheets using precomputed indices and save the processed maps.
        """
        for _, row in self.indices_df.iterrows():
            sheet = int(row['sheet'])
            start = int(row['start'])
            end = int(row['end'])

            if self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]:
                logging.info(f"Sheet {sheet} is empty. Skipping...")
                continue

            save_path = self.output_dir / f"delta-sheet-{sheet:02d}.fits"
            if save_path.exists() and not self.overwrite:
                logging.info(f"File {save_path} exists and overwrite is False. Skipping...")
                continue

            delta = self._get_mass_sheet(sheet, start, end)
            hp.write_map(str(save_path), delta, nest=True, dtype=np.float32)
            logging.info(f"Saved processed sheet {sheet} to {save_path}")

    def _get_mass_sheet(self, sheet: int, start: int, end: int) -> np.ndarray:
        """
        Process a mass sheet to compute the density contrast map.

        Args:
            sheet (int): Sheet index.
            start (int): Start index for data extraction.
            end (int): End index for data extraction.

        Returns:
            np.ndarray: Computed density contrast map.
        """
        logging.info(f"Processing sheet {sheet}: reading IDs and Mass values.")
        pid = self.msheets['ID'][start:end].compute()
        mass = self.msheets['Mass'][start:end].compute()
        ipix = pid % self.npix
        map_slice = np.bincount(ipix, weights=mass, minlength=self.npix)

        a1, a2 = self.aemit_index_edges[sheet:sheet + 2]
        z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
        chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
        volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * self.npix)
        delta = map_slice / (volume_diff * self.rhobar) - 1.0

        return delta