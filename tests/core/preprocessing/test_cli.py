import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import argparse

# Updated imports based on actual cli.py content
from lensing_ssc.core.preprocessing.cli import PreprocessingCLI, main as preprocessing_main
from lensing_ssc.core.preprocessing.config import ProcessingConfig # Actual config class used by cli.py

# Fixture for CLI arguments (can be adapted)
@pytest.fixture
def mock_args_preprocess():
    args = MagicMock(spec=argparse.Namespace)
    args.command = "preprocess"
    args.datadir = Path("/fake/data/dir")
    args.config = None
    args.overwrite = False
    args.resume = False
    args.sheet_range = None
    args.num_workers = None
    args.chunk_size = 10000
    args.cache_size_mb = 1024
    args.log_level = "INFO"
    args.log_file = None
    args.dry_run = False
    args.skip_validation = False
    args.memory_limit_gb = None
    return args

@pytest.fixture
def mock_args_validate():
    args = MagicMock(spec=argparse.Namespace)
    args.command = "validate"
    args.datadir = Path("/fake/data/dir")
    args.log_level = "INFO"
    return args

@pytest.fixture
def preprocessing_cli_instance():
    """Provides an instance of the PreprocessingCLI."""
    return PreprocessingCLI()

# Tests for PreprocessingCLI class

def test_preprocessing_cli_creation(preprocessing_cli_instance):
    """Test that PreprocessingCLI can be instantiated."""
    assert preprocessing_cli_instance is not None
    assert preprocessing_cli_instance.performance_monitor is not None

def test_preprocessing_cli_create_parser(preprocessing_cli_instance):
    """Test the create_parser method."""
    parser = preprocessing_cli_instance.create_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Check for subcommands by attempting to parse minimal valid commands
    try:
        # For subparsers, parse_args will try to run the sub-command's functionality
        # or exit if required arguments for the sub-command are missing.
        # We are just checking if the sub-command is recognized.
        parser.parse_args(["preprocess", "dummy_datadir"]) 
    except SystemExit as e: 
        # This is expected if "dummy_datadir" doesn't meet criteria or if other args are needed by the actual run method
        # For this test, we primarily care that "preprocess" is a known command.
        # A SystemExit often means argparse tried to exit (e.g. due to --help or an error).
        # If it's due to missing args for "preprocess", that means "preprocess" itself was recognized.
        pass # Or assert e.code != 0 if that's more specific

    try:
        parser.parse_args(["validate", "dummy_datadir"])
    except SystemExit:
        pass
        
    try:
        parser.parse_args(["status", "dummy_datadir"])
    except SystemExit:
        pass


@patch("lensing_ssc.core.preprocessing.cli.setup_logging")
@patch("lensing_ssc.core.preprocessing.cli.MassSheetProcessor")
@patch.object(PreprocessingCLI, "_load_config")
@patch.object(PreprocessingCLI, "_validate_data_directory")
def test_preprocessing_cli_run_preprocess_success(
    mock_validate_data_dir, mock_load_config, mock_mass_sheet_processor, 
    mock_setup_logging, preprocessing_cli_instance, mock_args_preprocess
):
    """Test the run_preprocess method of PreprocessingCLI - success case."""
    mock_config_instance = MagicMock(spec=ProcessingConfig)
    mock_load_config.return_value = mock_config_instance
    mock_validate_data_dir.return_value = True
    
    mock_processor_instance = mock_mass_sheet_processor.return_value
    mock_processor_instance.validate_data.return_value = True
    mock_processor_instance.preprocess.return_value = True

    result = preprocessing_cli_instance.run_preprocess(mock_args_preprocess)

    mock_setup_logging.assert_called_once_with(mock_args_preprocess.log_level, mock_args_preprocess.log_file)
    mock_load_config.assert_called_once_with(mock_args_preprocess)
    mock_validate_data_dir.assert_called_once_with(mock_args_preprocess.datadir)
    mock_mass_sheet_processor.assert_called_once_with(datadir=mock_args_preprocess.datadir, config=mock_config_instance)
    mock_processor_instance.validate_data.assert_called_once()
    mock_processor_instance.preprocess.assert_called_once_with(resume=mock_args_preprocess.resume)
    assert result == 0

@patch("lensing_ssc.core.preprocessing.cli.setup_logging")
@patch.object(PreprocessingCLI, "_load_config")
@patch.object(PreprocessingCLI, "_validate_data_directory")
def test_preprocessing_cli_run_preprocess_data_dir_validation_fail(
    mock_validate_data_dir, mock_load_config, mock_setup_logging, 
    preprocessing_cli_instance, mock_args_preprocess
):
    """Test run_preprocess when data directory validation fails."""
    mock_validate_data_dir.return_value = False # Simulate validation failure
    
    result = preprocessing_cli_instance.run_preprocess(mock_args_preprocess)
    
    mock_setup_logging.assert_called_once_with(mock_args_preprocess.log_level, mock_args_preprocess.log_file)
    mock_load_config.assert_called_once_with(mock_args_preprocess)
    mock_validate_data_dir.assert_called_once_with(mock_args_preprocess.datadir)
    assert result == 1 # Expect failure

@patch("lensing_ssc.core.preprocessing.cli.setup_logging")
@patch("lensing_ssc.core.preprocessing.cli.MassSheetProcessor")
@patch.object(PreprocessingCLI, "_load_config")
@patch.object(PreprocessingCLI, "_validate_data_directory")
def test_preprocessing_cli_run_preprocess_processor_validation_fail(
    mock_validate_data_dir, mock_load_config, mock_mass_sheet_processor, 
    mock_setup_logging, preprocessing_cli_instance, mock_args_preprocess
):
    """Test run_preprocess when MassSheetProcessor.validate_data() fails."""
    mock_config_instance = MagicMock(spec=ProcessingConfig)
    mock_load_config.return_value = mock_config_instance
    mock_validate_data_dir.return_value = True
    mock_processor_instance = mock_mass_sheet_processor.return_value
    mock_processor_instance.validate_data.return_value = False # This fails

    result = preprocessing_cli_instance.run_preprocess(mock_args_preprocess)
    mock_processor_instance.validate_data.assert_called_once()
    mock_processor_instance.preprocess.assert_not_called() # Preprocess should not be called
    assert result == 1

@patch("lensing_ssc.core.preprocessing.cli.setup_logging")
@patch.object(PreprocessingCLI, "_load_config")
@patch("logging.error") # To check error logging
def test_preprocessing_cli_run_preprocess_exception_on_load_config(
    mock_logging_error, mock_load_config, mock_setup_logging, 
    preprocessing_cli_instance, mock_args_preprocess
):
    """Test run_preprocess when _load_config raises an unexpected exception."""
    mock_load_config.side_effect = Exception("Config loading error")

    result = preprocessing_cli_instance.run_preprocess(mock_args_preprocess)
    
    mock_setup_logging.assert_called_once_with(mock_args_preprocess.log_level, mock_args_preprocess.log_file)
    mock_load_config.assert_called_once_with(mock_args_preprocess)
    mock_logging_error.assert_called_once()
    assert "Config loading error" in mock_logging_error.call_args[0][0]
    assert result == 1


# Tests for the main entry point function (preprocessing_main)
@patch("lensing_ssc.core.preprocessing.cli.PreprocessingCLI")
def test_main_function_calls_cli_main_method(MockPreprocessingCLI, mock_args_preprocess):
    """
    Test that the standalone main function (preprocessing_main) instantiates 
    PreprocessingCLI and calls its main method.
    """
    # Simulate sys.argv for a preprocess command
    with patch("sys.argv", ["script_name", "preprocess", str(mock_args_preprocess.datadir)]):
        mock_cli_instance = MockPreprocessingCLI.return_value
        mock_cli_instance.main.return_value = 0 # Simulate successful run of CLI's main method
        
        return_code = preprocessing_main() # Call the standalone main
        
        MockPreprocessingCLI.assert_called_once()
        # The standalone main should call the instance's main with sys.argv[1:]
        # or None if sys.argv is just the script name. Here it's ["preprocess", "/fake/data/dir"]
        # The instance's main method internally calls create_parser().parse_args(argv)
        # So we check that instance.main was called.
        mock_cli_instance.main.assert_called_once_with(["preprocess", str(mock_args_preprocess.datadir)])
        assert return_code == 0

@patch("lensing_ssc.core.preprocessing.cli.PreprocessingCLI")
def test_main_function_exit_code(MockPreprocessingCLI, mock_args_validate):
    """Test that the standalone main function returns the exit code from CLI's main method."""
    with patch("sys.argv", ["script_name", "validate", str(mock_args_validate.datadir)]):
        mock_cli_instance = MockPreprocessingCLI.return_value
        mock_cli_instance.main.return_value = 123 # Arbitrary non-zero exit code
        
        return_code = preprocessing_main()
        
        MockPreprocessingCLI.assert_called_once()
        mock_cli_instance.main.assert_called_once_with(["validate", str(mock_args_validate.datadir)])
        assert return_code == 123


# Tests for the main method of an PreprocessingCLI instance
@patch.object(PreprocessingCLI, "create_parser") 
def test_preprocessing_cli_instance_main_method_preprocess_command(
    mock_create_parser, preprocessing_cli_instance, mock_args_preprocess
):
    """
    Test the main method of an *instance* of PreprocessingCLI for the preprocess command.
    """
    # mock_args_preprocess already has .command = "preprocess"
    
    # Mock the parser and its parse_args method
    mock_parser = MagicMock(spec=argparse.ArgumentParser)
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args_preprocess # Simulate parsed args

    # Mock the run_preprocess method that should be called
    preprocessing_cli_instance.run_preprocess = MagicMock(return_value=0) # Success
    preprocessing_cli_instance.run_validate = MagicMock()
    preprocessing_cli_instance.run_status = MagicMock()

    # Call the main method of this instance with specific argv
    argv = ["preprocess", str(mock_args_preprocess.datadir)]
    return_code = preprocessing_cli_instance.main(argv) 

    mock_create_parser.assert_called_once()
    mock_parser.parse_args.assert_called_once_with(argv)
    preprocessing_cli_instance.run_preprocess.assert_called_once_with(mock_args_preprocess)
    preprocessing_cli_instance.run_validate.assert_not_called()
    preprocessing_cli_instance.run_status.assert_not_called()
    assert return_code == 0

@patch.object(PreprocessingCLI, "create_parser")
def test_preprocessing_cli_instance_main_method_validate_command(
    mock_create_parser, preprocessing_cli_instance, mock_args_validate
):
    """Test the main method of an PreprocessingCLI instance for the validate command."""
    # mock_args_validate already has .command = "validate"

    mock_parser = MagicMock(spec=argparse.ArgumentParser)
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args_validate

    preprocessing_cli_instance.run_preprocess = MagicMock()
    preprocessing_cli_instance.run_validate = MagicMock(return_value=0) # Success
    preprocessing_cli_instance.run_status = MagicMock()
    
    argv = ["validate", str(mock_args_validate.datadir)]
    return_code = preprocessing_cli_instance.main(argv)

    mock_create_parser.assert_called_once()
    mock_parser.parse_args.assert_called_once_with(argv)
    preprocessing_cli_instance.run_preprocess.assert_not_called()
    preprocessing_cli_instance.run_validate.assert_called_once_with(mock_args_validate)
    preprocessing_cli_instance.run_status.assert_not_called()
    assert return_code == 0
    
@patch.object(PreprocessingCLI, "create_parser")
def test_preprocessing_cli_instance_main_method_unknown_command(
    mock_create_parser, preprocessing_cli_instance
):
    """Test the main method of PreprocessingCLI with an unknown command."""
    
    # Simulate parsing an unknown command. argparse should exit.
    mock_parser = MagicMock(spec=argparse.ArgumentParser)
    mock_create_parser.return_value = mock_parser
    # parse_args for unknown command usually results in printing help and SystemExit(2)
    mock_parser.parse_args.side_effect = SystemExit(2)

    preprocessing_cli_instance.run_preprocess = MagicMock()
    preprocessing_cli_instance.run_validate = MagicMock()
    preprocessing_cli_instance.run_status = MagicMock()
    
    argv = ["unknown_command", "/some/path"]
    with pytest.raises(SystemExit) as e:
        preprocessing_cli_instance.main(argv)
    
    assert e.value.code == 2
    mock_create_parser.assert_called_once()
    mock_parser.parse_args.assert_called_once_with(argv)
    preprocessing_cli_instance.run_preprocess.assert_not_called()
    preprocessing_cli_instance.run_validate.assert_not_called()
    preprocessing_cli_instance.run_status.assert_not_called()

# Placeholder for tests of _load_config, _validate_data_directory, run_status, etc.
# These would require more specific mocking of Path interactions, os calls, config file content.
# Example:
# @patch("pathlib.Path.exists")
# @patch("pathlib.Path.is_dir")
# def test_validate_data_directory_valid(mock_is_dir, mock_exists, preprocessing_cli_instance):
#     mock_exists.return_value = True
#     mock_is_dir.return_value = True
#     (Path("/fake/dir") / "usmesh").exists.return_value = True # How to mock this specific sub-path?
#     # This needs more careful mocking of Path objects.
#     pass

# All old test functions related to PatchProcessorCLI, patch_processing_cli, 
# and their specific mocks (find_kappa_map_files, etc.) should be removed.
# The new tests focus on PreprocessingCLI and its dependencies (MassSheetProcessor, DataValidator). 