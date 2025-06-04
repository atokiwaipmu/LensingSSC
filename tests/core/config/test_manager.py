import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import json
import yaml
import toml

from lensing_ssc.core.config.manager import (
    ConfigManager,
    EnvironmentConfigManager,
    create_config_from_template,
    auto_detect_config_files,
    ConfigError,
)
from lensing_ssc.core.config.loader import (
    JSONConfigLoader,
    YAMLConfigLoader,
    TOMLConfigLoader,
    INIConfigLoader,
    UnsupportedFormatError,
    ConfigLoadError,
    ConfigSaveError,
)
from lensing_ssc.core.config.settings import ProcessingConfig # Assuming a base config class for templates

# Helper to create dummy config files
def create_dummy_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

@pytest.fixture
def temp_config_dir(tmp_path):
    return tmp_path

@pytest.fixture
def manager(temp_config_dir):
    # Provide a basic schema for testing validation; adjust as needed for actual ProcessingConfig
    schema = {
        "type": "object",
        "properties": {
            "setting1": {"type": "string"},
            "setting2": {"type": "number"},
            "nested": {
                "type": "object",
                "properties": {"sub1": {"type": "boolean"}}
            }
        },
        "required": ["setting1"]
    }
    return ConfigManager(
        default_config_path=str(temp_config_dir / "default.json"),
        schema=schema
    )

@pytest.fixture
def env_manager(temp_config_dir):
    schema = {
        "type": "object",
        "properties": {
            "setting1": {"type": "string"},
            "setting2": {"type": "number"},
        },
    }
    return EnvironmentConfigManager(
        env_prefix="LSSC_",
        default_config_path=str(temp_config_dir / "default_env.json"),
        schema=schema
    )

class TestConfigManager:
    sample_data = {"setting1": "value1", "setting2": 123, "nested": {"sub1": True}}
    sample_json = json.dumps(sample_data)
    sample_yaml = yaml.dump(sample_data)
    sample_toml = toml.dumps(sample_data)

    def test_load_config_json(self, manager, temp_config_dir):
        fpath = temp_config_dir / "test.json"
        create_dummy_file(fpath, self.sample_json)
        cfg = manager.load_config(str(fpath))
        assert cfg["setting1"] == "value1"
        assert cfg["setting2"] == 123

    def test_load_config_yaml(self, manager, temp_config_dir):
        fpath = temp_config_dir / "test.yaml"
        create_dummy_file(fpath, self.sample_yaml)
        cfg = manager.load_config(str(fpath))
        assert cfg["setting1"] == "value1"
        assert cfg["setting2"] == 123

    def test_load_config_toml(self, manager, temp_config_dir):
        fpath = temp_config_dir / "test.toml"
        create_dummy_file(fpath, self.sample_toml)
        cfg = manager.load_config(str(fpath))
        assert cfg["setting1"] == "value1"
        assert cfg["setting2"] == 123

    def test_load_config_caching(self, manager, temp_config_dir):
        fpath = temp_config_dir / "cache_test.json"
        create_dummy_file(fpath, self.sample_json)

        # First load, should not be cached
        cfg1 = manager.load_config(str(fpath), use_cache=True)
        assert manager.get_cache_info()[str(fpath)]['hits'] == 0

        # Second load, should be cached
        cfg2 = manager.load_config(str(fpath), use_cache=True)
        assert manager.get_cache_info()[str(fpath)]['hits'] == 1
        assert cfg1 == cfg2

        # Load without cache
        cfg3 = manager.load_config(str(fpath), use_cache=False)
        assert manager.get_cache_info()[str(fpath)]['hits'] == 1 # Hits shouldn't change

        manager.clear_cache()
        assert not manager.get_cache_info()

    def test_load_config_validation_success(self, manager, temp_config_dir):
        valid_data = {"setting1": "valid", "setting2": 10}
        fpath = temp_config_dir / "valid.json"
        create_dummy_file(fpath, json.dumps(valid_data))
        cfg = manager.load_config(str(fpath), validate=True)
        assert cfg["setting1"] == "valid"

    def test_load_config_validation_failure(self, manager, temp_config_dir):
        invalid_data = {"setting2": "should_be_number"} # Missing required 'setting1'
        fpath = temp_config_dir / "invalid.json"
        create_dummy_file(fpath, json.dumps(invalid_data))
        with pytest.raises(ConfigError): # Expecting ConfigError due to validation failure
            manager.load_config(str(fpath), validate=True)

    def test_load_config_file_not_found(self, manager):
        with pytest.raises(ConfigError): # Wraps ConfigLoadError
            manager.load_config("non_existent.json")

    def test_save_config(self, manager, temp_config_dir):
        fpath = temp_config_dir / "save_test.json"
        manager.save_config(self.sample_data, str(fpath))
        assert fpath.exists()
        with open(fpath, "r") as f:
            loaded_data = json.load(f)
        assert loaded_data == self.sample_data

    def test_save_config_overwrite(self, manager, temp_config_dir):
        fpath = temp_config_dir / "overwrite_test.json"
        create_dummy_file(fpath, json.dumps({"initial": "data"}))

        # Overwrite=False (default)
        with pytest.raises(ConfigError): # Wraps ConfigSaveError
             manager.save_config(self.sample_data, str(fpath), overwrite=False)

        # Overwrite=True
        manager.save_config(self.sample_data, str(fpath), overwrite=True)
        with open(fpath, "r") as f:
            loaded_data = json.load(f)
        assert loaded_data == self.sample_data

    def test_merge_configs(self, manager):
        base_cfg = {"setting1": "base", "setting2": 1, "nested": {"sub1": True}}
        override_cfg = {"setting2": 2, "nested": {"sub1": False, "sub2": "new"}}
        merged = manager.merge_configs(base_cfg, override_cfg)
        assert merged["setting1"] == "base"
        assert merged["setting2"] == 2
        assert merged["nested"]["sub1"] is False
        assert merged["nested"]["sub2"] == "new"

    def test_load_multiple_configs(self, manager, temp_config_dir):
        path1 = temp_config_dir / "multi1.json"
        path2 = temp_config_dir / "multi2.yaml"
        create_dummy_file(path1, json.dumps({"setting1": "val1", "setting2": 10}))
        create_dummy_file(path2, yaml.dump({"setting2": 20, "nested": {"sub1": True}}))

        cfg = manager.load_multiple_configs([str(path1), str(path2)])
        assert cfg["setting1"] == "val1"
        assert cfg["setting2"] == 20 # from path2, overriding path1
        assert cfg["nested"]["sub1"] is True

    def test_create_default_config_no_template(self, manager, temp_config_dir):
        # This test assumes create_default_config might save an empty dict or similar
        # if no template/schema implies wide open data. Or it might raise error.
        # Based on current manager init, it has a default_config_path.
        # If the default file doesn't exist, this might create it.
        default_path = Path(manager.default_config_path)
        if default_path.exists(): default_path.unlink()

        # Create a dummy schema if manager doesn't have one for this test
        if manager.schema is None:
            manager.schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        manager.create_default_config(overwrite=True) # path is manager.default_config_path
        assert default_path.exists()
        # Content check depends on implementation, e.g. empty dict or from schema defaults
        # For this test, just existence is checked.
        # with open(default_path, "r") as f:
        #     data = json.load(f)
        # assert data == {} or data == {"setting1": None} # Example assertion

    @patch('lensing_ssc.core.config.manager.validate') # Mock jsonschema.validate
    def test_validate_config_file(self, mock_validate, manager, temp_config_dir):
        fpath = temp_config_dir / "validate_me.json"
        create_dummy_file(fpath, self.sample_json)

        is_valid, errors = manager.validate_config_file(str(fpath))
        mock_validate.assert_called_once()
        # Actual validation depends on the schema provided to ConfigManager
        # Here we mostly test that the validation machinery is called.

    def test_get_config_template(self, manager):
        # This test depends on how ProcessingConfig is structured or if manager.schema is used
        # For now, assume it returns the schema if no specific template class is given
        template = manager.get_config_template()
        assert template == manager.schema

        # If a specific template class is used:
        # template_instance = manager.get_config_template(template_class=ProcessingConfig)
        # assert isinstance(template_instance, dict) # Or whatever ProcessingConfig.to_dict() returns
        # assert "data_dir" in template_instance # Example field from ProcessingConfig

    def test_get_cache_info_and_clear_cache(self, manager, temp_config_dir):
        fpath = temp_config_dir / "cache_info.json"
        create_dummy_file(fpath, self.sample_json)

        assert not manager.get_cache_info()
        manager.load_config(str(fpath), use_cache=True)
        cache_info = manager.get_cache_info()
        assert str(fpath) in cache_info
        assert cache_info[str(fpath)]['size'] > 0

        manager.clear_cache(str(fpath))
        assert str(fpath) not in manager.get_cache_info()

        manager.load_config(str(fpath), use_cache=True) # reload
        manager.clear_cache() # clear all
        assert not manager.get_cache_info()


class TestEnvironmentConfigManager:
    @patch.dict(os.environ, {"LSSC_SETTING1": "env_value1", "LSSC_SETTING2": "999"})
    def test_load_from_environment(self, env_manager):
        cfg = env_manager.load_from_environment()
        assert cfg["setting1"] == "env_value1"
        assert cfg["setting2"] == 999 # Assuming type conversion happens

    @patch.dict(os.environ, {}) # Clear relevant env vars
    def test_load_from_environment_with_defaults(self, env_manager, temp_config_dir):
        default_env_data = {"setting1": "default_env", "setting2": 111}
        create_dummy_file(env_manager.default_config_path, json.dumps(default_env_data))

        # Set default config path for the manager instance
        env_manager.config_path = env_manager.default_config_path

        cfg = env_manager.load_from_environment() # Should load defaults if env vars not set
        assert cfg["setting1"] == "default_env"
        assert cfg["setting2"] == 111

    @patch.dict(os.environ, {"LSSC_SETTING1": "env_override"})
    def test_get_environment_overrides(self, env_manager):
        overrides = env_manager.get_environment_overrides()
        assert overrides == {"setting1": "env_override"}

    @patch.dict(os.environ, {"LSSC_SETTING1": "to_be_cleared"})
    def test_clear_environment_overrides(self, env_manager):
        # This test is tricky because clear_environment_overrides might not be a feature.
        # EnvironmentConfigManager usually just reads from os.environ.
        # If it had its own internal override storage, this test would be different.
        # As is, os.environ needs to be manipulated directly for testing.
        env_manager.load_from_environment() # Load once
        os.environ.pop("LSSC_SETTING1", None) # Clear manually
        cfg_after_clear = env_manager.load_from_environment()
        assert "setting1" not in cfg_after_clear # Or it falls back to default

    @patch.dict(os.environ, {})
    def test_set_environment_defaults(self, env_manager):
        defaults = {"LSSC_SETTING1": "default_val", "LSSC_SETTING2": "123"}
        env_manager.set_environment_defaults(defaults)
        # Verify that os.environ now contains these, if not already set
        assert os.environ["LSSC_SETTING1"] == "default_val"
        assert os.environ["LSSC_SETTING2"] == "123"
        # Clean up
        del os.environ["LSSC_SETTING1"]
        del os.environ["LSSC_SETTING2"]


def test_create_config_from_template():
    # Assuming ProcessingConfig is a Pydantic model or dataclass
    # and can be instantiated and then converted to a dict.
    # This is a simplified example.
    class MockTemplateClass:
        def __init__(self, setting1="default", setting2=0):
            self.setting1 = setting1
            self.setting2 = setting2
        def dict(self): # if pydantic-like
            return {"setting1": self.setting1, "setting2": self.setting2}
        def to_dict(self): # if custom
             return {"setting1": self.setting1, "setting2": self.setting2}


    cfg_dict = create_config_from_template(MockTemplateClass)
    assert cfg_dict == {"setting1": "default", "setting2": 0}

    cfg_dict_override = create_config_from_template(MockTemplateClass, setting1="override")
    assert cfg_dict_override == {"setting1": "override", "setting2": 0}


def test_auto_detect_config_files(temp_config_dir):
    create_dummy_file(temp_config_dir / "config.json", "{}")
    create_dummy_file(temp_config_dir / "my_settings.yaml", "{}")
    create_dummy_file(temp_config_dir / "another.ini", "[]")
    create_dummy_file(temp_config_dir / "sub" / "deep_config.toml", "{}")

    # Test without specific names
    found_files = auto_detect_config_files(str(temp_config_dir))
    assert len(found_files) >= 4 # Could be more if other tests left files

    # Test with specific names
    found_specific = auto_detect_config_files(str(temp_config_dir), ["config.json", "my_settings.yaml"])
    assert len(found_specific) == 2
    assert str(temp_config_dir / "config.json") in [str(p) for p in found_specific]
    assert str(temp_config_dir / "my_settings.yaml") in [str(p) for p in found_specific]

    # Test with a common name
    found_common = auto_detect_config_files(str(temp_config_dir), ["config"])
    assert str(temp_config_dir / "config.json") in [str(p) for p in found_common]
    assert str(temp_config_dir / "sub" / "deep_config.toml") in [str(p) for p in found_common]

    # Test with depth
    found_deep = auto_detect_config_files(str(temp_config_dir / "sub"), search_depth=0) # Only current dir
    assert len(found_deep) == 1
    assert str(temp_config_dir / "sub" / "deep_config.toml") in [str(p) for p in found_deep]

    found_none_deep = auto_detect_config_files(str(temp_config_dir), search_depth=0) # only top
    assert str(temp_config_dir / "sub" / "deep_config.toml") not in [str(p) for p in found_none_deep]
    assert len(found_none_deep) == 3
