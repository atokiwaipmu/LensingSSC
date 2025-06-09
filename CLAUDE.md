# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LensingSSC is a Python package for studying Super-Sample Covariance (SSC) effects in weak gravitational lensing simulations. It provides tools for data preprocessing, statistical analysis, and lightcone-based simulations to compare different simulation setups (large boxes vs tiled small boxes).

## Development Commands

### Setup and Installation
```bash
# Create conda environment 
conda create -n lensingssc python=3.8
conda activate lensingssc

# Install with development dependencies
pip install -e .[dev]

# Alternative: Install dependencies separately
conda install -c bccp nbodykit
conda install numpy healpy matplotlib astropy scipy h5py pyyaml pandas
pip install lenstools classy cobaya
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lensing_ssc

# Run specific test types
pytest -m "not slow"           # Skip slow tests
pytest -m unit                 # Unit tests only  
pytest -m integration          # Integration tests only
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy lensing_ssc

# Linting
flake8 lensing_ssc

# All quality checks
black . && isort . && mypy lensing_ssc && flake8 lensing_ssc
```

### Main Analysis Pipeline
```bash
# Full analysis workflow (in order)
python scripts/01_run_preprocessing.py
python scripts/02_run_kappa_generation.py  
python scripts/03_run_analysis.py
python scripts/04_visualize_results.py

# Individual analysis runs
python -m lensing_ssc.run.run_patch_analysis --box_type tiled --zs 1.0 --overwrite
python -m lensing_ssc.run.run_patch_analysis --box_type bigbox --zs 2.0
python -m lensing_ssc.run.run_patch_noise
python -m lensing_ssc.run.run_stats_merge
```

## Architecture Overview

### Current Status: Modern Modular Architecture (70% Complete)

The codebase has undergone significant refactoring with a modern, dependency-abstracted architecture:

### Core Processing Pipeline
1. **Raw Simulation Data** → **Delta Sheets** (mass overdensity maps)
2. **Delta Sheets** → **Kappa Maps** (convergence maps via lensing theory)  
3. **Kappa Maps** → **Patches** (via Fibonacci grid sampling)
4. **Patches** → **Statistical Analysis** (multiple statistics per patch)
5. **Results** → **HDF5 Files** (for comparative analysis)

### ✅ Implemented Modern Architecture

**`lensing_ssc.core.interfaces/`** - Dependency abstraction layer
- `data_interface.py`, `compute_interface.py`, `storage_interface.py`, `plotting_interface.py`
- Enables lightweight core with heavy dependency injection

**`lensing_ssc.core.providers/`** - Provider pattern for heavy dependencies
- `factory.py` - Provider registry and factory system
- `healpix_provider.py`, `lenstools_provider.py`, `nbodykit_provider.py`, `matplotlib_provider.py`
- Lazy loading and dependency abstraction

**`lensing_ssc.core.processing/`** - Enhanced pipeline architecture
- `pipeline/` - Base pipeline classes with preprocessing and analysis workflows
- `steps/` - Modular processing steps (data loading, patching, statistics, output)
- `managers/` - Comprehensive management (resource, cache, checkpoint, progress, log, workflow)

**`lensing_ssc.core.base/`** - Lightweight core components
- `data_structures.py`, `coordinates.py`, `validation.py`, `exceptions.py`
- Independent of heavy astronomical libraries

**`lensing_ssc.core.config/`** - Centralized configuration management
- `settings.py`, `loader.py`, `manager.py` - YAML-based configuration with validation

### ⚠️ Legacy Components (To Be Reorganized)
**`lensing_ssc.core.fibonacci/` & `lensing_ssc.core.patch/`** - Original implementation
- Still functional but not following new modular architecture
- Will be migrated to `geometry/` and proper pipeline steps

### 🔄 Current Mixed State
- **New architecture** used for core infrastructure and processing
- **Legacy components** still used for scientific computations
- **Provider system** enables gradual migration without breaking functionality

### Configuration System

All analysis controlled via YAML files in `configs/`:
- `patch_size`: Patch size in degrees (default: 10)
- `nside`: HEALPix resolution (default: 8192) 
- `lmin`/`lmax`: Multipole range (300-3000)
- `zs_list`: Source redshifts [0.5, 1.0, 1.5, 2.0, 2.5]
- `ngal_list`: Galaxy densities for noise simulation [0, 7, 15, 30, 50]
- `sl_list`: Smoothing lengths in arcminutes [2, 5, 8, 10]

## Development Standards & Architecture Notes

### Code Quality
- **Type Hints:** Enforced via mypy with strict configuration
- **Formatting:** Black with 88-character line length
- **Import Sorting:** isort with Black-compatible profile
- **Testing:** pytest with coverage reporting, markers for test types (unit, integration, slow)

### Dependencies Structure
- **Core:** Scientific computing (numpy, scipy, matplotlib, astropy, pandas, h5py)
- **Heavy:** Astronomical packages (healpy, lenstools, nbodykit, classy, cobaya)
- **Dev:** Testing and quality tools (pytest, black, mypy, flake8, pre-commit)

### Architecture Implementation Status

**✅ Fully Implemented:**
- Provider pattern with dependency injection
- Lightweight core interfaces and base classes
- Configuration management system
- Enhanced pipeline architecture with comprehensive managers

**⚠️ Partially Implemented:**
- API layer (only `client.py` exists, missing specialized APIs)
- I/O system (basic file handlers exist, missing readers/writers structure)

**❌ Not Yet Implemented:**
- Plugin architecture system
- Proper module organization (geometry/, visualization/ vs current structure)
- Complete migration from legacy utilities

### Working with the Current Architecture

**Using Providers:** Access heavy dependencies through factory system:
```python
from lensing_ssc.core.providers.factory import get_provider
healpix = get_provider('healpix')
```

**Configuration:** Use centralized config management:
```python
from lensing_ssc.core.config.manager import ConfigManager
config = ConfigManager.load_config('configs/default.yaml')
```

**Processing:** Use new pipeline architecture for data processing workflows

### Output Organization
Results in `output/{box_type}/stats/` as HDF5 files:
- Naming: `stats_zs{redshift}_s{seed}_oa{opening_angle}.h5`
- Contains all statistical measures for comparison analysis
- Box types: `tiled` (500 Mpc boxes) vs `bigbox` (5 Gpc boxes)

### HPC Integration
Shell scripts in `lensing_ssc/run/scripts/` for cluster job submission:
- `run_patch_analysis.sh` - PBS job arrays for all parameter combinations
- `run_patch_noise.sh` - Noise addition workflow
- Designed for SLURM/PBS batch systems with conda environment activation

### Development Roadmap
See `docs/developplan.md` for detailed progress status and next steps. Key remaining tasks:
1. Implement plugin architecture
2. Complete API layer modules
3. Reorganize legacy utilities into proper module structure
4. Enhance testing coverage for new architecture

## Development Best Practices

- Always create a new branch and check out to it if you apply any modifications to files.
- After working on a branch, remember to push it and make a pull request to develop branch. Then check out back to develop branch.

## Memories

- One file shouldn't exceed 500 lines. if do, its time to separate dependencies.