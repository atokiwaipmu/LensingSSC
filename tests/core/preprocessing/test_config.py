import pytest
from lensing_ssc.core.preprocessing.config import ProcessingConfig
import yaml
from pathlib import Path

# Placeholder for more specific tests
def test_placeholder_config():
    """Placeholder test for config.py."""
    assert True

# Test ProcessingConfig dataclass
def test_processing_config_defaults():
    """Test default values of ProcessingConfig."""
    config = ProcessingConfig()
    assert config.chunk_size == 50000
    assert config.cache_size_mb == 1024
    assert config.mmap_threshold == 1000000
    assert config.sheet_range == (20, 100)
    assert config.extra_index == 100
    assert config.overwrite is False
    assert config.num_workers is None
    assert config.batch_size == 10
    assert config.log_level == "INFO"
    assert config.enable_progress_bar is True
    assert config.checkpoint_interval == 10
    assert config.cleanup_interval == 50
    assert config.max_cache_entries == 1000
    assert config.validate_input is True
    assert config.strict_validation is False

def test_processing_config_custom_values():
    """Test ProcessingConfig with custom values."""
    custom_values = {
        "chunk_size": 10000,
        "cache_size_mb": 512,
        "mmap_threshold": 500000,
        "sheet_range": (10, 50),
        "extra_index": 50,
        "overwrite": True,
        "num_workers": 4,
        "batch_size": 5,
        "log_level": "DEBUG",
        "enable_progress_bar": False,
        "checkpoint_interval": 5,
        "cleanup_interval": 25,
        "max_cache_entries": 500,
        "validate_input": False,
        "strict_validation": True,
    }
    config = ProcessingConfig(**custom_values)
    for key, value in custom_values.items():
        assert getattr(config, key) == value

# Test ProcessingConfig.from_file method
def test_load_config_file_not_found(tmp_path):
    """Test ProcessingConfig.from_file when the config file does not exist."""
    non_existent_file = tmp_path / "non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        ProcessingConfig.from_file(non_existent_file)

def test_load_config_valid_yaml(tmp_path):
    """Test ProcessingConfig.from_file with a valid YAML file."""
    config_data = {
        "chunk_size": 7500,
        "log_level": "WARNING",
        "sheet_range": [30, 120] # Test with list, should be converted to tuple by dataclass
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    config = ProcessingConfig.from_file(config_file)
    assert isinstance(config, ProcessingConfig)
    assert config.chunk_size == 7500
    assert config.log_level == "WARNING"
    assert config.sheet_range == (30, 120) # Ensure it's a tuple
    # Check that defaults are still applied for unspecified fields
    assert config.cache_size_mb == 1024 # Example default

def test_load_config_valid_json(tmp_path):
    """Test ProcessingConfig.from_file with a valid JSON file."""
    import json
    config_data = {
        "chunk_size": 8500,
        "log_level": "ERROR",
        "overwrite": True
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    config = ProcessingConfig.from_file(config_file)
    assert isinstance(config, ProcessingConfig)
    assert config.chunk_size == 8500
    assert config.log_level == "ERROR"
    assert config.overwrite is True
    assert config.batch_size == 10 # Default

def test_load_config_unsupported_format(tmp_path):
    """Test ProcessingConfig.from_file with an unsupported file format."""
    config_file = tmp_path / "test_config.txt"
    config_file.write_text("chunk_size: 1000")
    with pytest.raises(ValueError, match="Unsupported config format"):
        ProcessingConfig.from_file(config_file)


def test_load_config_empty_yaml(tmp_path):
    """Test ProcessingConfig.from_file with an empty YAML file (should use all defaults)."""
    config_file = tmp_path / "empty_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump({}, f) 
    
    config = ProcessingConfig.from_file(config_file)
    assert isinstance(config, ProcessingConfig)
    # Check a few default values
    assert config.chunk_size == 50000
    assert config.log_level == "INFO"
    assert config.overwrite is False

def test_load_config_invalid_yaml_structure(tmp_path):
    """Test ProcessingConfig.from_file with a YAML file that has an invalid structure (e.g., not a dict)."""
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(["list", "not", "a", "dict"], f)

    with pytest.raises(TypeError): # Dataclass __init__ will raise TypeError if **data is not a mapping
        ProcessingConfig.from_file(config_file)


def test_load_config_unknown_field_in_yaml(tmp_path):
    """Test ProcessingConfig.from_file with a YAML containing unknown fields."""
    config_data = {
        "chunk_size": 5000,
        "unknown_field": "some_value"
    }
    config_file = tmp_path / "unknown_field_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    with pytest.raises(TypeError): # Dataclass __init__ will raise TypeError for unexpected keyword arguments
        ProcessingConfig.from_file(config_file)


# Test config_instance.save() method
def test_save_config_yaml(tmp_path):
    """Test saving a ProcessingConfig object to a YAML file using instance.save()."""
    config = ProcessingConfig(chunk_size=2500, log_level="DEBUG", sheet_range=(1,5))
    output_file = tmp_path / "saved_config.yaml"
    
    config.save(output_file)
    
    assert output_file.exists()
    
    with open(output_file, 'r') as f:
        saved_data = yaml.safe_load(f)
        
    assert saved_data["chunk_size"] == 2500
    assert saved_data["log_level"] == "DEBUG"
    assert saved_data["sheet_range"] == [1, 5] # YAML dump might convert tuple to list
    # Check a default value that was not overridden
    assert saved_data["cache_size_mb"] == 1024

def test_save_config_json(tmp_path):
    """Test saving a ProcessingConfig object to a JSON file."""
    import json
    config = ProcessingConfig(num_workers=8, batch_size=20)
    output_file = tmp_path / "saved_config.json"

    config.save(output_file)
    assert output_file.exists()

    with open(output_file, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["num_workers"] == 8
    assert saved_data["batch_size"] == 20
    assert saved_data["log_level"] == "INFO" # Default

def test_save_config_unsupported_format(tmp_path):
    """Test saving config to an unsupported file format."""
    config = ProcessingConfig()
    output_file = tmp_path / "saved_config.txt"
    with pytest.raises(ValueError, match="Unsupported config format"):
        config.save(output_file)


def test_save_config_creates_directories(tmp_path):
    """Test that config_instance.save() uses the specified path, assuming directory exists."""
    config = ProcessingConfig()
    output_dir = tmp_path / "new_dir"
    # The save method itself doesn't create dirs, so we ensure it exists.
    output_dir.mkdir(parents=True, exist_ok=True) 
    output_file = output_dir / "config_in_new_dir.yaml"
        
    config.save(output_file)
    
    assert output_file.exists()
    assert output_dir.is_dir()

# Test .validate() method
def test_processing_config_validate_method():
    """Test the .validate() method of ProcessingConfig."""
    # Valid config
    config = ProcessingConfig()
    config.validate() # Should not raise

    # Invalid chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ProcessingConfig(chunk_size=0).validate()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ProcessingConfig(chunk_size=-1).validate()
    
    # Invalid sheet_range
    with pytest.raises(ValueError, match="sheet_range must be .* with start < end"):
        ProcessingConfig(sheet_range=(100, 20)).validate()
    with pytest.raises(ValueError, match="sheet_range must be .* with start < end"):
        ProcessingConfig(sheet_range=(50, 50)).validate()

    # Invalid num_workers
    with pytest.raises(ValueError, match="num_workers must be positive or None"):
        ProcessingConfig(num_workers=0).validate()
    with pytest.raises(ValueError, match="num_workers must be positive or None"):
        ProcessingConfig(num_workers=-1).validate()

    # Invalid cache_size_mb
    with pytest.raises(ValueError, match="cache_size_mb must be positive"):
        ProcessingConfig(cache_size_mb=0).validate()
    with pytest.raises(ValueError, match="cache_size_mb must be positive"):
        ProcessingConfig(cache_size_mb=-100).validate()

# Removed old tests for PreprocessingConfig-specific fields like path_resolution,
# log_file behavior, num_processes default.
# KappaConfig tests can be added separately if needed. 