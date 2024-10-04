
import logging
from pathlib import Path
from typing import Optional, Tuple
import os

import pandas as pd
import numpy as np
import healpy as hp
from nbodykit.lab import BigFileCatalog
from astropy.cosmology import FlatLambdaCDM

class IndicesFinder:
    """Finds and saves processing indices for mass sheets."""

    def __init__(self, datadir: Path, seed: int, extra_index: int = 100) -> None:
        """
        Initialize the IndicesFinder with data directory and parameters.

        Parameters:
            datadir (Path): Directory containing the mass sheets data.
            seed (int): Seed identifier for naming the indices CSV file.
            extra_index (int, optional): Extra index for processing. Defaults to 100.
        """
        self.datadir = datadir
        self.seed = seed
        self.extra_index = extra_index

        self.msheets = BigFileCatalog(
            self.datadir / "usmesh",
            dataset="HEALPIX/"
        )
        self.seed = self.msheets.attrs['seed'][0]
        self.aemit_index_offset = self.msheets.attrs['aemitIndex.offset']
        self.aemit_index_edges = self.msheets.attrs['aemitIndex.edges']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]

        self.save_path = self.datadir / f"preproc_s{self.seed}_indices.csv"

    def find_indices(self, i_start: int = 20, i_end: int = 100) -> None:
        """
        Find and save the start and end indices for each mass sheet.

        Parameters:
            i_start (int, optional): Starting sheet index. Defaults to 20.
            i_end (int, optional): Ending sheet index. Defaults to 100.
        """
        indices = []
        prev_end: Optional[int] = None

        for i in range(i_start, i_end):
            if self._is_sheet_empty(i):
                logging.info(f"Sheet {i} is empty. Skipping...")
                continue

            start, end = self._find_index(i, start=prev_end)
            prev_end = end
            indices.append({"sheet": i, "start": start, "end": end})

        if indices:
            indices_df = pd.DataFrame(indices)
            indices_df.to_csv(self.save_path, index=False)
            logging.info(f"Indices saved to {self.save_path}")
        else:
            logging.warning("No indices were found to save.")

    def _is_sheet_empty(self, sheet: int) -> bool:
        """
        Check if a given sheet is empty.

        Parameters:
            sheet (int): Sheet index to check.

        Returns:
            bool: True if the sheet is empty, False otherwise.
        """
        return self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]

    def _find_index(
        self,
        sheet: int,
        start: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Determine the start and end indices for processing a mass sheet.

        Parameters:
            sheet (int): Sheet index.
            start (Optional[int], optional): Previous end index. Defaults to None.

        Returns:
            Tuple[int, int]: Start and end indices for the sheet.
        """
        if start is not None:
            end = self.aemit_index_offset[sheet + 2]
        else:
            start = self.aemit_index_offset[sheet + 1]
            end = self.aemit_index_offset[sheet + 2]

            if self.extra_index:
                search_start = min(start + self.extra_index, end)
                aemit_start = self.msheets['Aemit'][start:search_start].compute()
                change_idx = np.where(np.diff(aemit_start) == self.a_interval)[0]

                if change_idx.size > 0:
                    delta = change_idx[0]
                    logging.info(
                        f"Aemit start changed from {start} to {start + delta}"
                    )
                    start += delta

        if start == end:
            return start, end

        logging.info(f"Index start: {start}, end: {end}")

        if self.extra_index:
            search_end = max(end - self.extra_index, start)
            aemit_end = self.msheets['Aemit'][search_end:end].compute()
            change_idx = np.where(np.round(np.diff(aemit_end), 2) == self.a_interval)[0]

            if change_idx.size > 0:
                delta = change_idx[0]
                logging.info(
                    f"Aemit end changed from {end} to {end - delta}"
                )
                end -= delta

        return start, end


class MassSheetProcessor:
    """Processes mass sheets for cosmological data visualization."""

    def __init__(
        self,
        datadir: Path,
        overwrite: bool = False
    ) -> None:
        """
        Initialize the mass sheet processor with cosmology settings and the mass sheets catalog.

        Parameters:
            datadir (Path): Directory containing the mass sheets data.
            seed (int): Seed identifier for precomputed indices CSV.
            extra_index (int, optional): Extra index for processing. Defaults to 100.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        """
        self.datadir = datadir
        self.overwrite = overwrite

        self.msheets = BigFileCatalog(
            self.datadir / "usmesh",
            dataset="HEALPIX/"
        )
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)

        self._initialize_output_directory()
        self._initialize_attributes()
        self.indices_df = self._load_precomputed_indices()

    def _initialize_output_directory(self) -> None:
        """Create the output directory for processed mass sheets."""
        self.output_dir = self.datadir / "mass_sheets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_attributes(self) -> None:
        """Initialize attributes from the mass sheets catalog."""
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
        Load precomputed start and end indices from a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing sheet, start, and end columns.
        """
        csv_path = self.datadir / f"preproc_s{self.seed}_indices.csv"
        if not csv_path.is_file():
            logging.error(f"Precomputed indices file not found: {csv_path}, run IndicesFinder")
            indices_finder = IndicesFinder(self.datadir, self.seed)
            indices_finder.find_indices()
        df = pd.read_csv(csv_path)
        expected_columns = {'sheet', 'start', 'end'}
        if not expected_columns.issubset(df.columns):
            logging.error(f"CSV file must contain columns: {expected_columns}")
            raise ValueError(f"CSV file must contain columns: {expected_columns}")
        return df
    
    def preprocess(self) -> None:
        """
        Preprocess the mass sheets and save the processed data to the output directory.
        Utilizes precomputed start and end indices from a CSV file.
        """
        for _, row in self.indices_df.iterrows():
            sheet = row['sheet']
            start, end = row['start'], row['end']

            # Check if the sheet is empty
            if self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]:
                logging.info(f"Sheet {sheet} is empty. Skipping...")
                continue

            save_path = self.output_dir / f"delta-sheet-{int(sheet):02d}.fits"

            if save_path.exists() and not self.overwrite:
                logging.info(f"Sheet {sheet} already exists. Skipping...")
                continue

            # Process and save the mass sheet
            delta = self._get_mass_sheet(int(sheet), int(start), int(end))
            hp.write_map(str(save_path), delta, nest=True, dtype=np.float32)
            logging.info(f"Saved delta sheet {sheet} to {save_path}")

    def _get_mass_sheet(
        self,
        sheet: int,
        start: int,
        end: int
    ) -> np.ndarray:
        """
        Process the mass sheet data for a given sheet index.

        Parameters:
            sheet (int): Sheet index.
            start (int): Start index for data extraction.
            end (int): End index for data extraction.

        Returns:
            np.ndarray: Processed density contrast map.
        """
        logging.info(f"Processing sheet {sheet}: reading ID and Mass")

        pid = self.msheets['ID'][start:end].compute()
        mass = self.msheets['Mass'][start:end].compute()
        ipix = pid % self.npix
        map_slice = np.bincount(ipix, weights=mass, minlength=self.npix)

        a1, a2 = self.aemit_index_edges[sheet:sheet + 2]
        z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
        chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
        volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * self.npix)
        delta = map_slice / volume_diff / self.rhobar - 1.0

        return delta