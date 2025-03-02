import logging
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from nbodykit.lab import BigFileCatalog

class IndicesFinder:
    """Class to find and save processing indices for mass sheets data."""

    def __init__(self, datadir: Path, seed: int, extra_index: int = 100) -> None:
        """
        Initialize the IndicesFinder.

        Args:
            datadir (Path): Directory containing the mass sheets data.
            seed (int): Seed identifier for naming the indices CSV file.
            extra_index (int, optional): Extra index for processing. Defaults to 100.
        """
        self.datadir = datadir
        self.initial_seed = seed  # original seed provided
        self.extra_index = extra_index

        self.msheets = BigFileCatalog(self.datadir / "usmesh", dataset="HEALPIX/")
        # Use the seed from the dataset if available
        self.seed = self.msheets.attrs.get('seed', [seed])[0]
        self.aemit_index_offset = self.msheets.attrs['aemitIndex.offset']
        self.aemit_index_edges = self.msheets.attrs['aemitIndex.edges']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]

        self.save_path = self.datadir / f"preproc_s{self.seed}_indices.csv"

    def find_indices(self, i_start: int = 20, i_end: int = 100) -> None:
        """
        Find and save the start and end indices for each mass sheet.

        Args:
            i_start (int, optional): Starting sheet index. Defaults to 20.
            i_end (int, optional): Ending sheet index. Defaults to 100.
        """
        indices: List[dict] = []
        prev_end: Optional[int] = None

        for i in range(i_start, i_end):
            if self.is_sheet_empty(i):
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

    def is_sheet_empty(self, sheet: int) -> bool:
        """
        Check if the given sheet is empty.

        Args:
            sheet (int): Sheet index to check.

        Returns:
            bool: True if the sheet is empty, otherwise False.
        """
        return self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]

    def _find_index(self, sheet: int, start: Optional[int] = None) -> Tuple[int, int]:
        """
        Determine the start and end indices for processing a mass sheet.

        Args:
            sheet (int): Sheet index.
            start (Optional[int], optional): Previous end index. Defaults to None.

        Returns:
            Tuple[int, int]: Tuple containing start and end indices.
        """
        if start is not None:
            start_index = start
            end_index = self.aemit_index_offset[sheet + 2]
        else:
            start_index = self.aemit_index_offset[sheet + 1]
            end_index = self.aemit_index_offset[sheet + 2]

            if self.extra_index:
                search_start = min(start_index + self.extra_index, end_index)
                aemit_slice = self.msheets['Aemit'][start_index:search_start].compute()
                diff = np.diff(aemit_slice)
                change_indices = np.where(diff == self.a_interval)[0]
                if change_indices.size > 0:
                    delta = change_indices[0]
                    logging.info(f"Aemit start changed from {start_index} to {start_index + delta}")
                    start_index += delta

        if start_index == end_index:
            return start_index, end_index

        logging.info(f"Determined indices - start: {start_index}, end: {end_index}")

        if self.extra_index:
            search_end = max(end_index - self.extra_index, start_index)
            aemit_slice_end = self.msheets['Aemit'][search_end:end_index].compute()
            diff_end = np.round(np.diff(aemit_slice_end), 2)
            change_indices_end = np.where(diff_end == self.a_interval)[0]
            if change_indices_end.size > 0:
                delta = change_indices_end[0]
                logging.info(f"Aemit end changed from {end_index} to {end_index - delta}")
                end_index -= delta

        return start_index, end_index