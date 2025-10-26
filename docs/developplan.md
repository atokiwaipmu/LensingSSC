### Development Plan

## Progress Status Overview

**Current Status:** ~70% Complete (Phases 1-2 Implemented, Phase 3 Pending)

### ✅ Completed Phases

## Phase 1: Dependency Refactoring & Core Architecture ✅ COMPLETE

### 1.1 ✅ Create Dependency Abstraction Layer - IMPLEMENTED
```python
# lensing_ssc/core/interfaces/ - ✅ COMPLETE
├── data_interface.py      # Abstract interfaces for data access
├── compute_interface.py   # Abstract interfaces for computations
├── storage_interface.py   # Abstract interfaces for storage
└── plotting_interface.py  # Abstract interfaces for plotting
```

### 1.2 ✅ Implement Lightweight Core - IMPLEMENTED
```python
# lensing_ssc/core/ - ✅ COMPLETE + ENHANCED
├── base/
│   ├── __init__.py
│   ├── data_structures.py  # Core data structures (independent of healpy/lenstools)
│   ├── coordinates.py      # Coordinate transformations (minimal numpy)
│   ├── validation.py       # Data validation utilities
│   └── exceptions.py       # ✅ Additional: Custom exceptions
├── math/
│   ├── __init__.py
│   ├── statistics.py       # Basic statistical functions
│   ├── transforms.py       # Mathematical transformations
│   └── interpolation.py    # Interpolation utilities
└── config/
    ├── __init__.py
    ├── settings.py         # Centralized configuration
    ├── loader.py           # Configuration loading
    └── manager.py          # ✅ Additional: Configuration management
```

## Phase 2: Modular Implementation ✅ COMPLETE

### 2.1 ✅ Implement Provider Pattern for Heavy Dependencies - IMPLEMENTED + ENHANCED
```python
# lensing_ssc/core/providers/ - ✅ COMPLETE + ENHANCED
├── __init__.py
├── base_provider.py        # ✅ Additional: Base provider class
├── healpix_provider.py     # Healpy abstraction
├── lenstools_provider.py   # Lenstools abstraction
├── nbodykit_provider.py    # Nbodykit abstraction
├── matplotlib_provider.py # ✅ Additional: Matplotlib abstraction
└── factory.py              # Provider factory + registry system
```

### 2.2 ✅ Refactor Processing Pipeline - IMPLEMENTED + ENHANCED
```python
# lensing_ssc/core/processing/ - ✅ COMPLETE + ENHANCED
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── base_pipeline.py    # Abstract pipeline class
│   ├── preprocessing.py    # Preprocessing pipeline
│   └── analysis.py         # Analysis pipeline
├── steps/
│   ├── __init__.py
│   ├── data_loading.py     # Individual processing steps
│   ├── patching.py
│   ├── statistics.py
│   └── output.py
└── managers/              # ✅ ENHANCED: 6 managers vs planned 2
    ├── __init__.py
    ├── resource_manager.py  # Memory/CPU management
    ├── checkpoint_manager.py # Checkpoint management
    ├── cache_manager.py     # ✅ Additional: Cache management
    ├── progress_manager.py  # ✅ Additional: Progress tracking
    ├── log_manager.py       # ✅ Additional: Logging management
    └── workflow_manager.py  # ✅ Additional: Workflow coordination
```

### 🚧 Remaining Tasks

## Phase 3: Enhanced Modularity ⏳ IN PROGRESS

### 3.1 ❌ Plugin Architecture - NOT IMPLEMENTED
```python
# lensing_ssc/plugins/ - ❌ MISSING
├── __init__.py
├── base_plugin.py          # Plugin interface
├── statistics/             # Statistics plugins
├── visualization/          # Visualization plugins
└── export/                 # Export format plugins
```

### 3.2 ⚠️ Clean API Layer - PARTIALLY IMPLEMENTED
```python
# lensing_ssc/api/ - ⚠️ PARTIAL
├── __init__.py
├── client.py              # ✅ Main client interface
├── preprocessing.py       # ❌ MISSING: Preprocessing API
├── analysis.py           # ❌ MISSING: Analysis API
└── visualization.py      # ❌ MISSING: Visualization API
```

### 3.3 ❌ Missing Modular Organization
```python
# lensing_ssc/geometry/ - ❌ MISSING (content exists in core/)
├── __init__.py
├── fibonacci.py           # Currently: core/fibonacci_utils.py
├── patching.py           # Currently: core/patching_utils.py
└── projections.py

# lensing_ssc/io/ - ⚠️ PARTIAL (simplified structure exists)
├── __init__.py
├── readers/              # ❌ MISSING (only file_handlers.py exists)
└── writers/              # ❌ MISSING

# lensing_ssc/visualization/ - ❌ MISSING (plotting/ exists instead)
├── __init__.py
├── base/
└── specialized modules
```

## Refactored File Structure

```
LensingSSC/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml         # Modern Python packaging
├── requirements/          # Organized requirements
│   ├── base.txt          # Core dependencies
│   ├── heavy.txt         # Heavy optional dependencies
│   ├── dev.txt           # Development dependencies
│   └── docs.txt          # Documentation dependencies
│
├── configs/              # Configuration files
│   ├── default.yaml
│   ├── environments/     # Environment-specific configs
│   └── schemas/          # Configuration schemas
│
├── data/                 # Project data (gitignored)
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/                 # Documentation
│   ├── api/
│   ├── tutorials/
│   └── examples/
│
├── notebooks/            # Jupyter notebooks
│   ├── examples/
│   └── research/
│
├── results/              # Output from analyses
│   ├── figures/
│   ├── tables/
│   └── stats_data/
│
├── scripts/              # Entry point scripts
│   ├── 01_run_preprocessing.py
│   ├── 02_run_kappa_generation.py
│   ├── 03_run_analysis.py
│   └── 04_visualize_results.py
│
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── lensing_ssc/          # Main package
    ├── __init__.py
    │
    ├── api/              # Public API layer
    │   ├── __init__.py
    │   ├── client.py
    │   ├── preprocessing.py
    │   ├── analysis.py
    │   └── visualization.py
    │
    ├── core/             # Core functionality (minimal dependencies)
    │   ├── __init__.py
    │   ├── base/         # Basic data structures and utilities
    │   │   ├── __init__.py
    │   │   ├── data_structures.py
    │   │   ├── coordinates.py
    │   │   ├── validation.py
    │   │   └── exceptions.py
    │   ├── math/         # Mathematical utilities
    │   │   ├── __init__.py
    │   │   ├── statistics.py
    │   │   ├── transforms.py
    │   │   └── interpolation.py
    │   ├── config/       # Configuration management
    │   │   ├── __init__.py
    │   │   ├── settings.py
    │   │   ├── validators.py
    │   │   └── loader.py
    │   └── interfaces/   # Abstract interfaces
    │       ├── __init__.py
    │       ├── data_interface.py
    │       ├── compute_interface.py
    │       ├── storage_interface.py
    │       └── plotting_interface.py
    │
    ├── providers/        # Heavy dependency abstractions
    │   ├── __init__.py
    │   ├── base_provider.py
    │   ├── healpix_provider.py
    │   ├── lenstools_provider.py
    │   ├── nbodykit_provider.py
    │   └── factory.py
    │
    ├── processing/       # Processing pipelines
    │   ├── __init__.py
    │   ├── pipeline/
    │   │   ├── __init__.py
    │   │   ├── base_pipeline.py
    │   │   ├── preprocessing.py
    │   │   └── analysis.py
    │   ├── steps/
    │   │   ├── __init__.py
    │   │   ├── data_loading.py
    │   │   ├── patching.py
    │   │   ├── statistics.py
    │   │   └── output.py
    │   └── managers/
    │       ├── __init__.py
    │       ├── resource_manager.py
    │       └── checkpoint_manager.py
    │
    ├── statistics/       # Statistical analysis
    │   ├── __init__.py
    │   ├── base/
    │   │   ├── __init__.py
    │   │   └── base_statistic.py
    │   ├── power_spectrum.py
    │   ├── bispectrum.py
    │   ├── pdf_analysis.py
    │   ├── peak_analysis.py
    │   └── correlation_analysis.py
    │
    ├── geometry/         # Geometric operations
    │   ├── __init__.py
    │   ├── fibonacci.py  # Fibonacci grid utilities
    │   ├── patching.py   # Patch extraction
    │   └── projections.py
    │
    ├── io/               # Input/output operations
    │   ├── __init__.py
    │   ├── readers/
    │   │   ├── __init__.py
    │   │   ├── fits_reader.py
    │   │   ├── hdf5_reader.py
    │   │   └── csv_reader.py
    │   ├── writers/
    │   │   ├── __init__.py
    │   │   ├── fits_writer.py
    │   │   ├── hdf5_writer.py
    │   │   └── csv_writer.py
    │   └── utils/
    │       ├── __init__.py
    │       └── path_utils.py
    │
    ├── visualization/    # Plotting and visualization
    │   ├── __init__.py
    │   ├── base/
    │   │   ├── __init__.py
    │   │   └── base_plotter.py
    │   ├── statistics_plots.py
    │   ├── correlation_plots.py
    │   ├── comparison_plots.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── styling.py
    │       └── layout.py
    │
    ├── theory/           # Theoretical models
    │   ├── __init__.py
    │   ├── ssc_models.py
    │   ├── cosmology.py
    │   └── predictions.py
    │
    ├── plugins/          # Plugin system
    │   ├── __init__.py
    │   ├── base_plugin.py
    │   ├── statistics/
    │   ├── visualization/
    │   └── export/
    │
    └── utils/            # General utilities
        ├── __init__.py
        ├── logging_utils.py
        ├── performance.py
        ├── constants.py
        └── helpers.py
```

## Key Refactoring Strategies

### 1. Dependency Injection Pattern
```python
# lensing_ssc/core/foundation/data_structures/__init__.py
from abc import ABC, abstractmethod
from typing import Optional, Any

class MapProvider(ABC):
    """Abstract interface for map operations"""
    
    @abstractmethod
    def read_map(self, path: str) -> Any:
        pass
    
    @abstractmethod
    def write_map(self, data: Any, path: str) -> None:
        pass

class ConvergenceMap:
    """Core convergence map class with injected dependencies"""
    
    def __init__(self, data: Any, provider: Optional[MapProvider] = None):
        self.data = data
        self._provider = provider or self._get_default_provider()
    
    def _get_default_provider(self) -> MapProvider:
        from lensing_ssc.providers.factory import get_map_provider
        return get_map_provider()
```

### 2. Lazy Loading Pattern
```python
# lensing_ssc/providers/healpix_provider.py
class HealpixProvider(MapProvider):
    """Lazy-loaded healpy provider"""
    
    def __init__(self):
        self._healpy = None
    
    @property
    def healpy(self):
        if self._healpy is None:
            try:
                import healpy as hp
                self._healpy = hp
            except ImportError:
                raise ImportError("healpy is required for this functionality")
        return self._healpy
    
    def read_map(self, path: str) -> Any:
        return self.healpy.read_map(path)
```

### 3. Configuration Management
```python
# lensing_ssc/core/config/settings.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class ProcessingConfig:
    """Centralized configuration for processing"""
    
    # Data paths
    data_dir: Path
    output_dir: Path
    
    # Processing parameters
    patch_size_deg: float = 10.0
    xsize: int = 2048
    num_workers: Optional[int] = None
    
    # Provider settings
    providers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = {
                'healpix': 'lensing_ssc.providers.healpix_provider.HealpixProvider',
                'lenstools': 'lensing_ssc.providers.lenstools_provider.LenstoolsProvider'
            }

class ConfigManager:
    """Configuration manager with validation"""
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> ProcessingConfig:
        # Implementation for loading from YAML/JSON
        pass
    
    @classmethod
    def validate_config(cls, config: ProcessingConfig) -> bool:
        # Implementation for config validation
        pass
```

### 4. Plugin Architecture
```python
# lensing_ssc/plugins/base_plugin.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        pass

class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
    
    def register(self, plugin: BasePlugin):
        self._plugins[plugin.name] = plugin
    
    def get_plugin(self, name: str) -> BasePlugin:
        return self._plugins.get(name)
```

## Updated Implementation Timeline

### ✅ Weeks 1-6: Foundation & Migration - COMPLETE
- [x] Create new file structure
- [x] Implement core interfaces and base classes
- [x] Create provider abstractions
- [x] Set up configuration management
- [x] Migrate existing functionality to new structure
- [x] Implement provider pattern for heavy dependencies
- [x] Create pipeline architecture
- [x] Update tests (partially)

### 🚧 Weeks 7-9: Enhancement - IN PROGRESS
- [ ] **Priority 1:** Implement plugin system architecture
- [ ] **Priority 2:** Complete API layer (preprocessing.py, analysis.py, visualization.py)
- [ ] **Priority 3:** Reorganize legacy utilities into proper modules (geometry/, io/, visualization/)
- [ ] **Priority 4:** Performance optimization
- [ ] **Priority 5:** Documentation update

### 📋 Next Phase: Testing & Finalization
- [ ] Comprehensive testing suite expansion
- [ ] Performance benchmarking against legacy implementation
- [ ] Migration guide for existing users
- [ ] Release preparation and version management

## Immediate Next Steps (Priority Order)

### 1. Complete Plugin Architecture
```bash
# Create missing plugin system
mkdir -p lensing_ssc/plugins/{statistics,visualization,export}
# Implement base_plugin.py and plugin registry
```

### 2. Reorganize Legacy Utilities
```bash
# Move utilities to proper modules
mkdir -p lensing_ssc/geometry
mv lensing_ssc/core/fibonacci_utils.py → lensing_ssc/geometry/fibonacci.py
mv lensing_ssc/core/patching_utils.py → lensing_ssc/geometry/patching.py
```

### 3. Complete API Layer
```bash
# Implement missing API modules
touch lensing_ssc/api/{preprocessing,analysis,visualization}.py
```

### 4. Enhance I/O Structure
```bash
# Create proper I/O organization
mkdir -p lensing_ssc/io/{readers,writers}
```