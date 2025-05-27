# LensingSSC

A Python package for studying Super-Sample Covariance (SSC) effects in weak gravitational lensing simulations.

This project provides tools to process raw cosmological simulation data into mass sheets (Healpix maps of density contrast) and subsequently analyze these for weak lensing studies, with a particular focus on understanding Super-Sample Covariance (SSC). It includes modules for data preprocessing, statistical analysis (e.g., power spectrum, bispectrum, peak counts), and visualization.

The core preprocessing pipeline efficiently converts particle data from simulations (e.g., from `usmesh` format) into mass maps suitable for lensing analysis. It handles large datasets, incorporates cosmological calculations via `astropy`, and utilizes `healpy` for map manipulations.

**TODO: Add further details if necessary, e.g., specific simulation types supported, unique analysis methods, or research goals.**

This repository contains code for ... (e.g., simulating and analyzing strong gravitational lensing effects on the cosmic microwave background, particularly focusing on the Sunyaev-Zel'dovich effect with super-sample covariance).

## Repository Structure

The repository is organized as follows:

- `lensing_ssc/`: Main source code directory.
    - `core/`: Core functionalities of the project.
        - `preprocessing/`: Code related to data preprocessing.
        - `preprocessing_utils.py`: Utility functions for preprocessing.
        - `fibonacci_utils.py`: Utilities related to Fibonacci sphere/pixelization.
        - `patching_utils.py`: Utilities for creating patches from maps.
    - `plotting/`: Scripts and modules for generating plots.
        - `bad_patch_plots.py`: Functions to plot or identify bad patches.
        - `correlation_plots.py`: Functions to plot correlation analyses.
        - `rmp_rip_plots.py`: Functions to plot RMP/RIP related quantities.
        - `statistics_plots.py`: Functions to plot various statistical measures.
        - `plot_utils.py`: General utility functions for plotting.
    - `utils/`: General utility functions and constants.
        - `extractors.py`: Functions for extracting specific data or features.
        - `constants.py`: Physical or numerical constants used in the project.
        - `theory/`: (Potentially theoretical utilities, content not fully clear from structure)
    - `io/`: Modules for handling input and output operations.
        - `file_handlers.py`: Functions for reading and writing various file formats.
    - `theory/`: Modules related to theoretical models and calculations.
        - `ssc_model.py`: (Likely contains the SSC model implementation).
    - `stats/`: Modules for statistical analysis.
        - `bispectrum.py`: Functions related to bispectrum calculations.
        - `pdf.py`: Functions related to Probability Density Functions.
        - `peak_counts.py`: Functions for peak counting statistics.
        - `power_spectrum.py`: Functions related to power spectrum calculations.
    - `legacy_notebook/`: (Likely contains old Jupyter notebooks).
    - `legacy_run/`: (Likely contains old run scripts).
    - `legacy_utils/`: (Likely contains old utility functions).
    - `legacy_core/`: (Likely contains old core functionalities).
    - `legacy_theory/`: (Likely contains old theoretical models).
    - `__init__.py`: Initializes the `lensing_ssc` package.

## Installation

It is recommended to set up a virtual environment.

1. **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/atokiwaipmu/LensingSSC.git
    cd LensingSSC
    ```

2. **Install dependencies:**

    You can install the required packages using `pip` and the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    The dependencies include:
    - numpy
    - healpy
    - matplotlib
    - astropy
    - lenstools
    - scipy
    - h5py
    - pyyaml
    - classy
    - pandas
    - nbodykit
    - cobaya

3. **Install the package:**

    To make the `LensingSSC` package available in your Python environment, run:
    ```bash
    python setup.py install
    ```
    Alternatively, for development purposes, you might prefer to install it in editable mode:
    ```bash
    python setup.py develop
    # or
    # pip install -e .
    ```

## Usage

**TODO: Add more detailed usage examples and explanations here.**

### Basic Usage: Running Scripts

The `scripts/` directory in the root of the repository contains several scripts that demonstrate common workflows. 
For example, to run the mass sheet preprocessing:

```bash
python scripts/01_run_preprocessing.py /path/to/your/data --log-level INFO
```
Replace `/path/to/your/data` with the actual path to your input data directory.
This script utilizes `lensing_ssc.core.preprocessing.MassSheetProcessor` to process raw simulation data.

Other available scripts include:
- `scripts/02_run_kappa_generation.py`: For generating convergence (kappa) maps.
- `scripts/03_run_analysis.py`: For performing various analyses on the generated data.
- `scripts/04_visualize_results.py` / `scripts/04_visualize_results_refactored.py`: For visualizing results.

Refer to each script's command-line help for more options (e.g., `python scripts/01_run_preprocessing.py --help`).

### Library Usage

Here are some examples of how to use the `LensingSSC` library components directly in your Python scripts or notebooks.

**1. Preprocessing Mass Sheets:**

This example shows how to use `MassSheetProcessor` to process raw data. You would typically point it to your data directory containing `usmesh` files.

```python
from pathlib import Path
from lensing_ssc.core.preprocessing import MassSheetProcessor, ProcessingConfig

# Configuration for preprocessing
config = ProcessingConfig(
    overwrite=False,  # Whether to overwrite existing processed files
    # sheet_range=(0, 10)  # Optional: process only a subset of sheets, e.g., sheets 0 to 9
)

# Path to your data directory
# This directory should contain a subdirectory named 'usmesh' with your raw data.
data_dir = Path("/path/to/your/simulation/output")

if not (data_dir / "usmesh").exists():
    print(f"Error: 'usmesh' subdirectory not found in {data_dir}")
else:
    # Initialize the processor
    processor = MassSheetProcessor(datadir=data_dir, config=config)

    # Run the preprocessing
    # This will generate .fits files in data_dir / "mass_sheets" / 
    results = processor.preprocess(resume=True) # resume=True allows resuming if interrupted
    print(f"Preprocessing results: {results}")
```

**2. Calculating Power Spectrum:**

This example demonstrates how to calculate the power spectrum from a convergence map (e.g., one derived from the processed mass sheets). You would first need to load or generate a `ConvergenceMap` object from `lenstools`.

```python
import numpy as np
import healpy as hp
from lenstools import ConvergenceMap # Assuming lenstools is installed and used for ConvergenceMap
from lensing_ssc.stats.power_spectrum import calculate_power_spectrum

# --- This part is an assumption of how you might get a ConvergenceMap ---
# TODO: Replace with actual code to load/create your ConvergenceMap
# Example: Load a processed mass sheet (delta map) and convert to ConvergenceMap
# This assumes your mass sheets are Healpix maps and you have a way to define the map angle.

# Placeholder: Load a Healpix map (e.g., a delta sheet)
try:
    delta_map = hp.read_map("/path/to/your/simulation/output/mass_sheets/delta-sheet-00.fits")
    nside = hp.get_nside(delta_map)
    map_angle = hp.nside2resol(nside, arcmin=True) * nside # A rough estimate for map_angle in arcmin
    # Create a ConvergenceMap object (parameters might vary based on your data)
    # You might need to convert your delta map to kappa first depending on your conventions.
    # Assuming delta_map can be directly used or is proportional to kappa for this example.
    kappa_map = ConvergenceMap(data=delta_map, angle=map_angle * 'arcmin')
except IOError:
    print("Error: Could not load example map. Please provide a valid path and ensure lenstools is set up.")
    kappa_map = None # Set to None if map loading fails
# --- End of assumed part ---

if kappa_map:
    # Define multipole (l) bins for power spectrum calculation
    l_min = 10
    l_max = hp.nside2lmax(nside) # Or a suitable l_max for your analysis
    num_bins = 20
    l_edges = np.logspace(np.log10(l_min), np.log10(l_max), num_bins + 1)
    # Calculate bin centers (can be geometric or arithmetic mean, or specific l values)
    ell_centers = (l_edges[:-1] + l_edges[1:]) / 2.0

    # Calculate the power spectrum
    # The function returns Cl * l * (l+1) / (2*pi)
    cl_data = calculate_power_spectrum(kappa_map, l_edges, ell_centers)

    print(f"Calculated Power Spectrum (l-centers): {ell_centers}")
    print(f"Calculated Power Spectrum (Cl_scaled): {cl_data}")

    # You can then plot ell_centers vs cl_data
else:
    print("Skipping power spectrum calculation as kappa_map was not loaded.")

```

**TODO: Add more examples for other modules like bispectrum, peak counts, plotting utilities etc.**

## Contributing

We welcome contributions to LensingSSC! If you'd like to contribute, please follow these guidelines:

### Reporting Bugs
- Check the GitHub Issues tracker to see if the bug has already been reported.
- If not, open a new issue. Please include a clear title and description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements
- Check the GitHub Issues tracker to see if the enhancement has already been suggested.
- If not, open a new issue. Please provide a clear title and a detailed description of the proposed enhancement and its potential benefits.

### Coding Standards
- **Style**: Please follow PEP 8 for Python code. Use a linter (like Flake8 or Pylint) to check your code.
- **Docstrings**: Write clear and concise docstrings for all modules, classes, functions, and methods. We recommend following the NumPy/SciPy docstring format or Google Python Style Guide for docstrings.
- **Type Hinting**: Use type hints for function signatures and variables where appropriate to improve code readability and maintainability (Python 3.8+).

### Testing
- **TODO**: Describe the testing framework used (e.g., pytest, unittest) and how to run tests.
- Write new tests for any new features or bug fixes.
- Ensure all tests pass before submitting a pull request.

### Pull Request Process
1. Fork the repository.
2. Create a new branch for your feature or bug fix (e.g., `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-123`).
3. Make your changes and commit them with clear, descriptive commit messages.
4. Push your branch to your fork (e.g., `git push origin feature/my-new-feature`).
5. Open a pull request against the `main` (or `develop`) branch of the original repository.
6. Clearly describe the changes in your pull request and link to any relevant issues.
7. Ensure your pull request passes any automated checks (CI/CD).
8. Be prepared to address any feedback or requested changes from the maintainers.

By contributing, you agree that your contributions will be licensed under the MIT License.

## License

This project is licensed under the MIT License. See the `LICENSE` file in the root directory for more details. 