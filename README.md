# LensingSSC

LensingSSC is a Python package for studying **Super-Sample Covariance (SSC)** effects in weak gravitational lensing simulations. It provides tools for data preprocessing, statistical analysis, and the generation of lightcone-based simulations. The code in this repository underpins the research presented in the paper *"Lensing Super Sample Covariance"* by Akira Tokiwa, Adrian E. Bayer, Jia Liu, and Masahiro Takada.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Statistical Quantities](#statistical-quantities)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Analysis](#results--analysis)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## Introduction

Super-Sample Covariance (SSC) represents the impact of modes larger than the survey area on measured statistics in weak lensing maps. LensingSSC is designed to:
- Preprocess and patch simulation data
- Perform higher-order statistical analysis
- Facilitate comparative studies between different simulation setups

This package is ideal for researchers and practitioners working with large-scale structure and weak lensing surveys.

---

## Features

- **Preprocessing Modules:**  
  Tools for lightcone patching and data preprocessing.
  
- **Statistical Analysis:**  
  Functions to compute higher-order statistics such as power spectra, bispectra, and probability density functions.
  
- **Example Notebooks:**  
  Jupyter notebooks that demonstrate package usage and typical workflows.
  
- **Unit Testing:**  
  Built-in tests to ensure accuracy and robustness.
  
- **Extensible Configuration:**  
  YAML-based configuration files for flexible customization of analysis parameters.

---

## Project Structure

```plaintext
LensingSSC/
├── .gitignore
├── LICENSE
├── README.md             # (Should be updated to reflect this new structure)
├── requirements.txt
├── setup.py              # Configured to recognize `lensing_ssc` as the main package
│
├── configs/              # YAML configuration files (e.g., default.yaml)
│
├── data/                 # (Typically Gitignored) All project-specific data
│   ├── raw/              # Downloaded/original simulation data (e.g., usmesh files)
│   ├── interim/          # Intermediate data products (e.g., mass_sheets/)
│   └── processed/        # Data ready for analysis (e.g., kappa_maps/)
│
├── docs/                 # Project documentation
│   ├── installation.md
│   ├── usage.md          # (Should be updated with new script commands)
│   └── api/
│
├── notebooks/            # Jupyter notebooks for exploration, examples, paper figures
│
├── results/              # Output from analyses and scripts
│   ├── figures/          # Generated plots and figures
│   ├── tables/           # Generated tables (e.g., CSV, LaTeX)
│   └── stats_data/       # Numerical statistical results (e.g., power spectra data files)
│
├── scripts/              # Main executable scripts for running the pipeline stages
│   ├── 01_run_preprocessing.py    # Handles mass sheet generation
│   ├── 02_run_kappa_generation.py # Handles kappa map generation
│   ├── 03_run_analysis.py         # Runs statistical analyses on kappa maps
│   └── 04_visualize_results.py    # Generates plots/tables from statistical results
│
├── lensing_ssc/          # Main Python package source code
│   ├── __init__.py
│   ├── core/             # Core algorithms (data manipulation, patching)
│   │   ├── __init__.py
│   │   ├── preprocessing_utils.py
│   │   └── patching_utils.py
│   ├── stats/            # Statistical measurement modules
│   │   ├── __init__.py
│   │   ├── power_spectrum.py
│   │   ├── bispectrum.py
│   │   ├── pdf.py
│   │   └── peak_counts.py
│   ├── theory/           # Theoretical models and calculations
│   │   ├── __init__.py
│   │   └── ssc_model.py (example)
│   ├── io/               # Data loading and saving utilities
│   │   ├── __init__.py
│   │   └── file_handlers.py
│   ├── plotting/         # Reusable plotting functions
│   │   ├── __init__.py
│   │   └── plot_utils.py
│   └── utils/            # General helper functions
│       ├── __init__.py
│       └── (e.g., healpix_helpers.py, config_loader.py)
│
└── tests/                # Unit and integration tests
    ├── __init__.py
    ├── test_core.py
    ├── test_stats.py
    └── (other test_*.py files)
```

*Each directory is organized to separate core functionalities, examples, tests, and documentation.*

---

## Statistical Quantities

LensingSSC calculates various statistical quantities on weak lensing maps, including:

- **Power Spectrum:**  
  Uses `lenstools.ConvergenceMap.powerSpectrum` to capture the power distribution across scales.
  
- **Squeezed Bispectrum:**  
  Computed via `lenstools.ConvergenceMap.bispectrum` for nonlinear structure analysis.
  
- **Probability Density Function (PDF):**  
  Analyzes the convergence map distribution using `lenstools.ConvergenceMap.pdf`.
  
- **Peak/Minima Statistics:**  
  Identifies features in the maps with `lenstools.ConvergenceMap.locatePeaks`.

---

## Installation

For complete installation details, please refer to [installation.md](installation.md).

<details>
  <summary><strong>Installation Instructions (Click to expand)</strong></summary>

  ### Prerequisites

  - **Python 3.8.19**  
    Download from the [official Python website](https://www.python.org/downloads/release/python-3819/) and verify with:
    ```sh
    python --version
    ```
    The output should be `Python 3.8.x`.

  ### Create a Conda Environment

  ```sh
  conda create -n lensingssc python=3.8
  conda activate lensingssc
  ```

  ### Installing Dependencies

  **Using Conda:**

  1. Install `nbodykit`:
     ```sh
     conda install -c bccp nbodykit
     ```
  2. Install additional packages:
     ```sh
     conda install numpy healpy matplotlib astropy scipy h5py pyyaml pandas
     ```

  **Using Pip:**

  1. Install Lenstools, Classy, and Cobaya:
     ```sh
     pip install lenstools classy cobaya
     ```
  2. Install CLASS via Cobaya:
     ```sh
     pip install cobaya --upgrade
     mkdir ./lib
     cobaya-install cosmo -p ./lib
     ```

  ### Installing LensingSSC

  1. **Clone the repository:**
     ```sh
     git clone https://github.com/atokiwaipmu/LensingSSC.git
     cd LensingSSC
     ```
  2. **Install the package:**
     ```sh
     python setup.py install
     ```

  ### Verification

  Confirm installation with:
  ```sh
  python -c "import lensing_ssc; print('LensingSSC installed successfully')"
  ```

</details>

---

## Data Download

Before running the analysis, you need to download the required data files. The data is publicly available and can be accessed from multiple sources.

<details>
  <summary><strong>Data Download Instructions (Click to expand)</strong></summary>

  The data is available from [HalfDome Simulations Data](https://halfdomesims.github.io/data/). You can download the files as follows:
  
  - **Globus Download:**  
    The data is publicly available on Globus. Follow the instructions on the website for a direct download.
  
  - **NERSC Community File Storage (CFS):**  
    The data is also hosted on the NERSC CFS system in the CMB project directory at:
    ```
    /global/cfs/cdirs/cmb/data/halfdome/
    ```
  
  Additional details:
  - **Description:**  
    For a full description of the data, see the enclosed `README.txt` provided with the data.
  
  - **Example Usage:**  
    For an example of reading the data, refer to the notebook `Halfdome_analysis.ipynb` included with the download.
  
  - **Future Updates:**  
    Further examples, notebooks, and additional data will be added soon!
  
</details>

---

## Usage

### Generating Data

- **Generate Mass Sheets:**
  ```sh
  python -m src.preproc /path/to/usmesh
  ```

- **Generate Kappa Maps:**
  ```sh
  python -m src.kappamap /path/to/mass_sheets
  ```

- **Perform Statistical Analysis on Kappa Maps:**
  ```sh
  python -m src.analysis_patch /path/to/kappa
  ```

### Configuration

Configuration is managed via YAML files located in the `configs` directory. Key parameters include:
- `patch_size`
- `nbin`
- `lmin`, `lmax`
- Other analysis-specific parameters

For optimal performance, the package uses **Fibonacci-patched full-sky maps**.

---

## Results & Analysis

LensingSSC supports comparative studies between different simulation setups:

- **Large Boxes (e.g., 5 Gpc):**  
  Captures all modes, including those contributing to SSC.

- **Tiled Small Boxes (e.g., 500 Mpc):**  
  May miss large-scale modes, affecting redshift-dependent SSC effects.

### Sample Plots

- **Correlation Matrix:**  
  ![Correlation Matrix](img/comparison/correlation_zs2.0_oa10_sl2_noiseless.png "Correlation Matrix for z_source=2.0")
  
- **Comparison of Mean Values:**  
  ![Mean Comparison](img/comparison/mean_zs2.0_oa10_sl2_noiseless.png "Mean Comparison for z_source=2.0")
  
- **Diagonal Covariance Terms:**  
  ![Diagonal Covariance](img/comparison/diagonal_zs2.0_oa10_sl2_noiseless.png "Diagonal terms of the covariance matrix for z_source=2.0")

---

## Contributing

Contributions are welcome! Please follow these guidelines:
- **Submit Issues:**  
  Report bugs or suggest enhancements via the GitHub issue tracker.
- **Pull Requests:**  
  Follow the established code style and include tests when adding new features.
- **Documentation:**  
  Ensure any changes include appropriate updates to the documentation.

For detailed guidelines, please see [CONTRIBUTING.md](CONTRIBUTING.md) (if available).

---

## References

- **Preprocessing for Lensing Maps:**  
  [preproc-kappa](https://github.com/HalfDomeSims/preproc-kappa.git)
- **Statistical Analysis:**  
  [HOS-Y1](https://github.com/LSSTDESC/HOS-Y1-prep.git)
- **Example Notebook:**  
  [CorrelatedSims](https://github.com/liuxx479/CorrelatedSims/blob/master/hack_crowncanyon_kappa.ipynb)
- **Job Submission Script:**  
  [sbatch_gen.py](https://github.com/liuxx479/CorrelatedSims/blob/master/sbatch_gen.py)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*For additional questions or support, please open an issue on GitHub or contact the project maintainers.*
