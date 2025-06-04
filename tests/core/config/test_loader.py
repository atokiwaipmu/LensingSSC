import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from lensing_ssc.core.config.loader import ( # Adjusted import
    ConfigLoader,
    JSONConfigLoader,
    YAMLConfigLoader,
    TOMLConfigLoader,
    INIConfigLoader,
    get_config_loader,
    get_available_loaders,
    get_loader_info,
    suggest_format,
    UnsupportedFormatError,
    ConfigLoadError,
    ConfigSaveError,
)

# Helper function to create dummy files
def create_dummy_file(filepath, content=""):
    dirname = os.path.dirname(filepath)
    if dirname: # Only call makedirs if there's a directory part
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

# Helper function to remove dummy files
def remove_dummy_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.fixture
def json_loader():
    return JSONConfigLoader()

@pytest.fixture
def yaml_loader():
    return YAMLConfigLoader()

@pytest.fixture
def toml_loader():
    return TOMLConfigLoader()

@pytest.fixture
def ini_loader():
    return INIConfigLoader()

class TestConfigLoaderSubclasses:
    # Test data
    valid_data = {"key": "value", "number": 123}
    invalid_json_content = "{'key': 'value', 'number': 123}" # Invalid JSON, uses single quotes
    valid_json_content = '{"key": "value", "number": 123}'
    valid_yaml_content = "key: value\nnumber: 123"
    invalid_yaml_content = "key: value\n  number: 123" # Invalid YAML, inconsistent indentation
    valid_toml_content = 'key = "value"\nnumber = 123'
    invalid_toml_content = 'key = "value"\nnumber = 123a' # Invalid TOML, invalid number
    valid_ini_content = "[section]\nkey = value\nnumber = 123"
    invalid_ini_content = "key = value\nnumber = 123" # Invalid INI, no section

    @pytest.mark.parametrize(
        "loader_fixture, valid_content, invalid_content, filename_ext",
        [
            ("json_loader", valid_json_content, invalid_json_content, "json"),
            ("yaml_loader", valid_yaml_content, invalid_yaml_content, "yaml"),
            ("toml_loader", valid_toml_content, invalid_toml_content, "toml"),
            ("ini_loader", valid_ini_content, invalid_ini_content, "ini"),
        ],
    )
    def test_load_valid_and_invalid_files(
        self, loader_fixture, valid_content, invalid_content, filename_ext, request
    ):
        loader = request.getfixturevalue(loader_fixture)
        valid_filepath = f"test_valid.{filename_ext}"
        invalid_filepath = f"test_invalid.{filename_ext}"

        # Test loading valid file
        create_dummy_file(valid_filepath, valid_content)
        loaded_data = loader.load(valid_filepath)
        if filename_ext == "ini": # INI files have sections
            assert loaded_data["section"]["key"] == "value"
            assert int(loaded_data["section"]["number"]) == 123
        else:
            assert loaded_data == self.valid_data
        remove_dummy_file(valid_filepath)

        # Test loading invalid file
        create_dummy_file(invalid_filepath, invalid_content)
        with pytest.raises(ConfigLoadError):
            loader.load(invalid_filepath)
        remove_dummy_file(invalid_filepath)

    @pytest.mark.parametrize(
        "loader_fixture, filename_ext",
        [
            ("json_loader", "json"),
            ("yaml_loader", "yaml"),
            ("toml_loader", "toml"),
            ("ini_loader", "ini"),
        ],
    )
    def test_save_and_verify_output(self, loader_fixture, filename_ext, request):
        loader = request.getfixturevalue(loader_fixture)
        filepath = f"test_save.{filename_ext}"

        if filename_ext == "ini":
            data_to_save = {"section": self.valid_data}
        else:
            data_to_save = self.valid_data

        loader.save(data_to_save, filepath)

        # Verify the saved content
        # For INI, the structure will be slightly different when loaded back
        # We will compare the content string for INI for simplicity here
        if filename_ext == "ini":
            with open(filepath, "r") as f:
                saved_content = f.read()
            # Normalize line endings and handle potential minor formatting differences
            expected_content_lines = sorted([line.strip() for line in self.valid_ini_content.strip().splitlines()])
            saved_content_lines = sorted([line.strip() for line in saved_content.strip().splitlines()])
            assert expected_content_lines == saved_content_lines
        else:
            loaded_data = loader.load(filepath)
            assert loaded_data == self.valid_data

        remove_dummy_file(filepath)


    @pytest.mark.parametrize(
        "loader_fixture", ["json_loader", "yaml_loader", "toml_loader", "ini_loader"]
    )
    def test_file_not_found_error(self, loader_fixture, request):
        loader = request.getfixturevalue(loader_fixture)
        with pytest.raises(ConfigLoadError):
            loader.load("non_existent_file.txt")

    @pytest.mark.parametrize(
        "loader_fixture", ["json_loader", "yaml_loader", "toml_loader", "ini_loader"]
    )
    @patch("builtins.open", side_effect=PermissionError)
    def test_permission_error_on_load(self, mock_open_perm, loader_fixture, request):
        loader = request.getfixturevalue(loader_fixture)
        with pytest.raises(ConfigLoadError):
            loader.load("any_file.txt")

    @pytest.mark.parametrize(
        "loader_fixture", ["json_loader", "yaml_loader", "toml_loader", "ini_loader"]
    )
    @patch("builtins.open", side_effect=PermissionError)
    def test_permission_error_on_save(self, mock_open_perm, loader_fixture, request):
        loader = request.getfixturevalue(loader_fixture)
        with pytest.raises(ConfigSaveError):
            loader.save({"data": "test"}, "any_file.txt")


class TestHelperFunctions:
    def test_get_config_loader(self):
        assert isinstance(get_config_loader("file.json"), JSONConfigLoader)
        assert isinstance(get_config_loader("file.yaml"), YAMLConfigLoader)
        assert isinstance(get_config_loader("file.yml"), YAMLConfigLoader)
        assert isinstance(get_config_loader("file.toml"), TOMLConfigLoader)
        assert isinstance(get_config_loader("file.ini"), INIConfigLoader)
        with pytest.raises(UnsupportedFormatError):
            get_config_loader("file.txt")

    def test_get_available_loaders(self):
        loaders = get_available_loaders()
        assert "json" in loaders
        assert "yaml" in loaders
        assert "toml" in loaders
        assert "ini" in loaders
        assert len(loaders) == 4

    def test_get_loader_info(self):
        info = get_loader_info("json")
        assert info["name"] == "JSON"
        assert info["extensions"] == [".json"]

        info = get_loader_info("yaml")
        assert info["name"] == "YAML"
        assert info["extensions"] == [".yaml", ".yml"]

        info = get_loader_info("toml")
        assert info["name"] == "TOML"
        assert info["extensions"] == [".toml"]

        info = get_loader_info("ini")
        assert info["name"] == "INI"
        assert info["extensions"] == [".ini"]

        with pytest.raises(UnsupportedFormatError):
            get_loader_info("txt")

    def test_suggest_format(self):
        assert suggest_format({"key": "value"}) == "json" # Default for simple dicts
        assert suggest_format([1, 2, 3]) == "json" # Default for lists
        assert suggest_format({"key": "value", "nested": {"sub_key": "sub_value"}}) == "json"
        # TOML might be suggested for more complex nested structures if logic is added
        # For now, it will default to JSON as per current simple implementation.
        # If a more sophisticated suggestion logic is implemented in core.config.loader,
        # this test would need to be updated.

        # Test with a Path object for filename hint
        assert suggest_format({"key": "value"}, filename=Path("config.toml")) == "toml"
        assert suggest_format({"key": "value"}, filename=Path("config.yaml")) == "yaml"
        assert suggest_format({"key": "value"}, filename=Path("config.JSON")) == "json" # case-insensitivity
        assert suggest_format({"key": "value"}, filename=Path("backup.config.ini")) == "ini"
        assert suggest_format({"key": "value"}, filename=Path("unknown.ext")) == "json" # fallback

# Abstract methods test
class TestConfigLoaderAbstractMethods:
    def test_load_not_implemented(self):
        class TestLoader(ConfigLoader):
            name = "Test"
            extensions = [".test"]
            def save(self, data, filepath): # pragma: no cover
                pass

        loader = TestLoader()
        with pytest.raises(NotImplementedError):
            loader.load("file.test")

    def test_save_not_implemented(self):
        class TestLoader(ConfigLoader):
            name = "Test"
            extensions = [".test"]
            def load(self, filepath): # pragma: no cover
                pass

        loader = TestLoader()
        with pytest.raises(NotImplementedError):
            loader.save({}, "file.test")

# Test for ConfigLoader itself (though it's abstract, some parts can be tested)
def test_config_loader_base():
    assert ConfigLoader.name == "AbstractConfig" # Default name
    assert ConfigLoader.extensions == [] # Default extensions

    class ConcreteLoader(ConfigLoader):
        name = "Concrete"
        extensions = [".concrete"]
        def load(self, filepath): return {} # pragma: no cover
        def save(self, data, filepath): pass # pragma: no cover

    assert ConcreteLoader.name == "Concrete"
    assert ConcreteLoader.extensions == [".concrete"]

    # Test __str__ and __repr__
    loader = ConcreteLoader()
    assert str(loader) == "ConcreteLoader (handles: .concrete)"
    assert repr(loader) == "<ConcreteLoader (handles: .concrete)>"

    # Test get_info
    info = ConcreteLoader.get_info()
    assert info["name"] == "Concrete"
    assert info["extensions"] == [".concrete"]
    assert isinstance(info["loader_class"], type)
    assert info["loader_class"] == ConcreteLoader
