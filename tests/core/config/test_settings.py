import pytest
import os
from pathlib import Path
from unittest.mock import patch
import logging

from lensing_ssc.core.config.settings import (
    BaseConfig,
    ProcessingConfig,
    AnalysisConfig,
    VisualizationConfig,
    get_config,
    set_config,
    reset_config,
    update_config,
    load_config_from_env,
    CONFIG_SCHEMA, # Assuming this is the jsonschema for validation
    ConfigError,
)
from lensing_ssc.core.config.manager import ConfigManager # For testing global config fns context

# Disable all logging for tests to keep output clean
logging.disable(logging.CRITICAL)

# Helper for creating dummy files if needed by any config settings (e.g. path validation)
def create_dummy_file(filepath, content=""):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create dummy usmesh dir for ProcessingConfig validation
    (data_dir / "usmesh").mkdir()
    return data_dir

@pytest.fixture
def processing_config_data(temp_data_dir):
    return {
        "data_dir": str(temp_data_dir),
        "output_dir": str(temp_data_dir / "output"),
        "run_name": "test_run",
    }

@pytest.fixture
def analysis_config_data():
    return {
        "l_min": 10.0,
        "l_max": 2000.0,
        "n_l_bins": 20,
        "binning_scheme": "log",
    }

@pytest.fixture
def visualization_config_data():
    return {
        "style": "seaborn-v0_8-darkgrid",
        "dpi": 150,
        "output_format": "png",
    }

class TestProcessingConfig:
    def test_initialization_defaults(self, temp_data_dir):
        # Requires data_dir to be set for validation pass
        cfg = ProcessingConfig(data_dir=str(temp_data_dir))
        assert cfg.run_name == "default_run"
        assert cfg.log_level == "INFO"
        assert cfg.output_dir == Path.cwd() / "output" / "default_run" # Default construction
        assert cfg.overwrite is False
        assert cfg.resume is False
        assert cfg.healpix_nside == 4096
        # ... test other defaults

    def test_initialization_custom(self, processing_config_data, temp_data_dir):
        cfg = ProcessingConfig(**processing_config_data)
        assert cfg.data_dir == Path(processing_config_data["data_dir"])
        assert cfg.output_dir == Path(processing_config_data["output_dir"])
        assert cfg.run_name == processing_config_data["run_name"]

    def test_validation_invalid_data_dir(self, tmp_path):
        with pytest.raises(ConfigError): # Or Pydantic's ValidationError if used directly
            ProcessingConfig(data_dir=str(tmp_path / "non_existent_dir"))

    def test_validation_invalid_nside(self, temp_data_dir):
        with pytest.raises(ConfigError):
            ProcessingConfig(data_dir=str(temp_data_dir), healpix_nside=100) # Not power of 2

    def test_to_dict(self, processing_config_data):
        cfg = ProcessingConfig(**processing_config_data)
        cfg_dict = cfg.to_dict()
        assert cfg_dict["run_name"] == processing_config_data["run_name"]
        assert cfg_dict["data_dir"] == str(processing_config_data["data_dir"]) # Paths converted to str

    def test_from_dict(self, processing_config_data, temp_data_dir):
        cfg = ProcessingConfig.from_dict(processing_config_data)
        assert cfg.run_name == processing_config_data["run_name"]
        assert cfg.data_dir == Path(processing_config_data["data_dir"])

        # Test with partial data (should use defaults)
        partial_data = {"data_dir": str(temp_data_dir), "run_name": "partial_run"}
        cfg_partial = ProcessingConfig.from_dict(partial_data)
        assert cfg_partial.run_name == "partial_run"
        assert cfg_partial.log_level == "INFO" # Default

    def test_get_provider_class(self, processing_config_data):
        cfg = ProcessingConfig(**processing_config_data)
        # Assuming default provider map or some logic in get_provider_class
        # This test might need more specific setup if get_provider_class is complex
        # For example, if it dynamically loads, mocks might be needed.
        # If 'healpix' is a key in a predefined map:
        # assert cfg.get_provider_class("healpix") is not None
        with pytest.raises(ValueError): # If provider name is invalid
             cfg.get_provider_class("non_existent_provider")

        # Example for a specific provider
        cfg.providers['my_custom_healpix'] = 'lensing_ssc.core.providers.healpix_provider.HealpixProvider'
        provider_class = cfg.get_provider_class('my_custom_healpix')
        assert provider_class.__name__ == 'HealpixProvider'


class TestAnalysisConfig:
    def test_initialization_defaults(self):
        cfg = AnalysisConfig()
        assert cfg.l_min == 100.0
        assert cfg.n_patches == 1000
        # ... test other defaults

    def test_initialization_custom(self, analysis_config_data):
        cfg = AnalysisConfig(**analysis_config_data)
        assert cfg.l_min == analysis_config_data["l_min"]
        assert cfg.n_l_bins == analysis_config_data["n_l_bins"]

    def test_validation_invalid_l_range(self):
        with pytest.raises(ConfigError):
            AnalysisConfig(l_min=2000.0, l_max=100.0)

    def test_validation_invalid_binning(self):
        with pytest.raises(ConfigError):
            AnalysisConfig(binning_scheme="invalid_scheme")

    def test_to_dict(self, analysis_config_data):
        cfg = AnalysisConfig(**analysis_config_data)
        cfg_dict = cfg.to_dict()
        assert cfg_dict["l_min"] == analysis_config_data["l_min"]
        assert cfg_dict["binning_scheme"] == analysis_config_data["binning_scheme"]

    def test_from_dict(self, analysis_config_data):
        cfg = AnalysisConfig.from_dict(analysis_config_data)
        assert cfg.l_min == analysis_config_data["l_min"]
        assert cfg.n_l_bins == analysis_config_data["n_l_bins"]

    def test_get_l_bins(self, analysis_config_data):
        cfg = AnalysisConfig(**analysis_config_data)
        l_bins = cfg.get_l_bins()
        assert len(l_bins) == cfg.n_l_bins + 1
        assert l_bins[0] == pytest.approx(cfg.l_min)
        assert l_bins[-1] == pytest.approx(cfg.l_max)

        cfg_lin = AnalysisConfig(binning_scheme="linear", l_min=0, l_max=10, n_l_bins=10)
        l_bins_lin = cfg_lin.get_l_bins()
        assert np.allclose(l_bins_lin, np.linspace(0,10,11))


class TestVisualizationConfig:
    def test_initialization_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.output_format == "pdf"
        assert cfg.font_size == 12
        # ... test other defaults

    def test_initialization_custom(self, visualization_config_data):
        cfg = VisualizationConfig(**visualization_config_data)
        assert cfg.style == visualization_config_data["style"]
        assert cfg.dpi == visualization_config_data["dpi"]

    def test_validation_invalid_format(self):
        with pytest.raises(ConfigError):
            VisualizationConfig(output_format="invalid")

    def test_to_dict(self, visualization_config_data):
        cfg = VisualizationConfig(**visualization_config_data)
        cfg_dict = cfg.to_dict()
        assert cfg_dict["style"] == visualization_config_data["style"]
        assert cfg_dict["output_format"] == visualization_config_data["output_format"]

    def test_from_dict(self, visualization_config_data):
        cfg = VisualizationConfig.from_dict(visualization_config_data)
        assert cfg.style == visualization_config_data["style"]
        assert cfg.dpi == visualization_config_data["dpi"]


class TestGlobalConfigFunctions:
    @pytest.fixture(autouse=True)
    def reset_global_config_for_each_test(self, temp_data_dir):
        # This ensures that global state from one test doesn't affect another
        # We need a valid default path for reset_config() to work without error if it tries to load default
        dummy_default_path = temp_data_dir / "global_default.json"
        create_dummy_file(dummy_default_path, json.dumps({"data_dir": str(temp_data_dir)}))

        # Mock ConfigManager's default path to use this dummy for global tests
        # This is a bit indirect; ideally, global functions would allow path injection for testing
        with patch.object(ConfigManager, 'DEFAULT_CONFIG_PATH', str(dummy_default_path)):
            with patch.object(ConfigManager, 'USER_CONFIG_PATH', str(temp_data_dir / "user_config.json")):
                 with patch.object(ConfigManager, 'PROJECT_CONFIG_PATH', str(temp_data_dir / "project_config.json")):
                    reset_config() # Reset before each test
                    yield
                    reset_config() # Reset after each test

    def test_get_set_config(self, processing_config_data):
        cfg_initial = get_config() # Should be a default ProcessingConfig or similar
        assert isinstance(cfg_initial, ProcessingConfig)

        new_cfg_data = processing_config_data.copy()
        new_cfg_data["run_name"] = "global_set_test"
        new_processing_cfg = ProcessingConfig(**new_cfg_data)

        set_config(new_processing_cfg)
        cfg_after_set = get_config()
        assert cfg_after_set.run_name == "global_set_test"
        assert cfg_after_set.data_dir == Path(new_cfg_data["data_dir"])

    def test_reset_config(self, processing_config_data, temp_data_dir):
        # Set a custom config first
        custom_cfg_data = processing_config_data.copy()
        custom_cfg_data["run_name"] = "custom_for_reset"
        custom_processing_cfg = ProcessingConfig(**custom_cfg_data)
        set_config(custom_processing_cfg)
        assert get_config().run_name == "custom_for_reset"

        reset_config()
        cfg_after_reset = get_config()
        # This assertion depends on what reset_config() defaults to.
        # Assuming it reloads a default ProcessingConfig.
        # The default_run_name might be "default_run" or similar.
        # Also, data_dir would be from the mocked DEFAULT_CONFIG_PATH
        assert cfg_after_reset.run_name != "custom_for_reset"
        assert cfg_after_reset.data_dir == temp_data_dir # From dummy global_default.json

    def test_update_config(self, processing_config_data, temp_data_dir):
        # Ensure data_dir is valid for initial config
        initial_data = processing_config_data.copy()
        initial_data["run_name"] = "initial_update_run"
        initial_cfg = ProcessingConfig(**initial_data)
        set_config(initial_cfg)

        updates = {"run_name": "updated_run_name", "log_level": "DEBUG"}
        update_config(updates)

        cfg_after_update = get_config()
        assert cfg_after_update.run_name == "updated_run_name"
        assert cfg_after_update.log_level == "DEBUG"
        assert cfg_after_update.data_dir == Path(initial_data["data_dir"]) # Unchanged

    @patch.dict(os.environ, {
        "LSSC_RUN_NAME": "env_run",
        "LSSC_LOG_LEVEL": "WARNING",
        "LSSC_DATA_DIR": "/tmp/env_data_dir" # Example, may not be valid path for real run
    })
    @patch("pathlib.Path.exists", return_value=True) # Mock path existence for env data_dir
    @patch("lensing_ssc.core.config.settings.Path.is_dir", return_value=True)
    def test_load_config_from_env(self, mock_is_dir, mock_exists, temp_data_dir):
        # Create a dummy usmesh under the mocked env_data_dir for validation
        # This is tricky because os.environ is patched, not the actual file system for this part
        # We rely on patching Path.exists and is_dir for the env var path

        # To ensure the base config (before env override) is valid for data_dir:
        initial_cfg = ProcessingConfig(data_dir=str(temp_data_dir))
        set_config(initial_cfg)

        cfg = load_config_from_env(prefix="LSSC_")
        assert isinstance(cfg, ProcessingConfig)
        assert cfg.run_name == "env_run"
        assert cfg.log_level == "WARNING"
        assert str(cfg.data_dir) == "/tmp/env_data_dir"

    def test_get_config_type_switching(self, analysis_config_data):
        # Set a different type of config
        analysis_cfg = AnalysisConfig(**analysis_config_data)
        set_config(analysis_cfg)

        current_cfg = get_config()
        assert isinstance(current_cfg, AnalysisConfig)
        assert current_cfg.l_min == analysis_config_data["l_min"]

        # Reset should bring it back to default (likely ProcessingConfig)
        reset_config()
        default_cfg_after_reset = get_config()
        assert isinstance(default_cfg_after_reset, ProcessingConfig)

    def test_update_config_with_dict(self, processing_config_data):
        initial_cfg = ProcessingConfig(**processing_config_data)
        set_config(initial_cfg)

        update_dict = {"run_name": "dict_update", "healpix_nside": 2048}
        updated_cfg = update_config(update_dict) # update_config returns the modified config

        assert updated_cfg.run_name == "dict_update"
        assert updated_cfg.healpix_nside == 2048
        assert get_config().run_name == "dict_update" # Global state also updated

    def test_update_config_with_another_config_object(self, processing_config_data, temp_data_dir):
        cfg1_data = processing_config_data.copy()
        cfg1_data["run_name"] = "cfg1_run"
        cfg1 = ProcessingConfig(**cfg1_data)
        set_config(cfg1)

        cfg2_data = processing_config_data.copy() # Make sure data_dir is valid
        cfg2_data["data_dir"] = str(temp_data_dir / "data2") # Different data_dir
        (temp_data_dir / "data2").mkdir()
        (temp_data_dir / "data2" / "usmesh").mkdir()
        cfg2_data["run_name"] = "cfg2_run"
        cfg2_data["log_level"] = "DEBUG"
        cfg2 = ProcessingConfig(**cfg2_data)

        updated_cfg = update_config(cfg2)
        assert updated_cfg.run_name == "cfg2_run"
        assert updated_cfg.log_level == "DEBUG"
        assert updated_cfg.data_dir == Path(cfg2_data["data_dir"])
        assert get_config().run_name == "cfg2_run"

# Note: CONFIG_SCHEMA itself is not directly testable here unless we try to validate
# data against it using jsonschema library, which is implicitly done by ConfigManager tests.
# Testing BaseConfig might be limited as it's an abstract class,
# but its concrete subclasses are tested.
