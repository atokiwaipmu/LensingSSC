# Installation Guide for LensingSSC

This guide provides step-by-step instructions for installing the LensingSSC package using a conda environment.

## Prerequisites

### Python 3.8
Ensure you have Python 3.8 installed. Download it from the [official Python website](https://www.python.org/downloads/release/python-3819/).

Verify your Python version:
```sh
python --version
```
The output should be `Python 3.8.x`.

## Creating a Conda Environment

Create and activate a new conda environment with Python 3.8:
```sh
conda create -n lensingssc python=3.8
conda activate lensingssc
```

## Installing Dependencies

### Using Conda

1. **Install `nbodykit`:**
   ```sh
   conda install -c bccp nbodykit
   ```
2. **Install other available packages via conda:**
   ```sh
   conda install numpy healpy matplotlib astropy scipy h5py pyyaml pandas
   ```

### Using Pip

Some packages are best installed via pip:

1. **Install Lenstools, Classy, and Cobaya:**
   ```sh
   pip install lenstools classy cobaya
   ```
2. **Install CLASS via Cobaya:**
   ```sh
   pip install cobaya --upgrade
   mkdir ./lib
   cobaya-install cosmo -p ./lib
   ```

### Optional Backend Providers

LensingSSC supports additional backend providers for various functionalities. These often require extra dependencies that can be installed as needed.

**MySQL Provider (`mysql`)**

*   The `MySQLProvider` allows LensingSSC to use a MySQL database for certain storage operations (e.g., managing metadata or simulation outputs, if configured to do so).
*   **Dependency**: `mysql-connector-python`
*   **Installation**: To include this provider, install LensingSSC with the `mysql` extra:
    ```sh
    pip install lensing-ssc[mysql]
    ```
    If you have already installed LensingSSC, you can install the connector separately:
    ```sh
    pip install mysql-connector-python>=8.0.0
    ```
*   **Configuration**: When configuring providers in your LensingSSC setup, use the name `'mysql'` to refer to this provider.

## Installing LensingSSC

1. **Clone the repository:**
   ```sh
   git clone https://github.com/atokiwaipmu/LensingSSC.git
   cd LensingSSC
   ```
2. **Install the package:**
   ```sh
   python setup.py install
   ```

## Verification

Confirm the installation by running:
```sh
python -c "import lensing_ssc; print('LensingSSC installed successfully')"
```

## Troubleshooting

- Ensure you are using Python 3.8.
- Verify that your conda environment is activated.
- Re-run the installation commands if any errors occur.

For further assistance, please open an issue on the [GitHub repository](https://github.com/atokiwaipmu/LensingSSC/issues).

Happy Lensing!