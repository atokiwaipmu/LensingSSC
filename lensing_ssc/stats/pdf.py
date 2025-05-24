# lensing_ssc/stats/pdf.py
import numpy as np
from lenstools import ConvergenceMap

def calculate_pdf(conv_map: ConvergenceMap, bins: np.ndarray) -> np.ndarray:
    """
    Calculates the Probability Density Function (PDF) of a convergence map.

    Parameters
    ----------
    conv_map : ConvergenceMap
        The input convergence map. The map should ideally be smoothed and 
        normalized by its standard deviation if comparing with theoretical PDFs 
        or if specific bin ranges (like -4 to 4 sigma) are used.
    bins : np.ndarray
        The bin edges for the PDF histogram.

    Returns
    -------
    np.ndarray
        The values of the PDF at the center of the bins.
    """
    # The pdf method returns a tuple: (bin_centers, pdf_values)
    # We take the pdf_values part, which is at index 1.
    pdf_values = conv_map.pdf(bins)[1]
    return pdf_values 