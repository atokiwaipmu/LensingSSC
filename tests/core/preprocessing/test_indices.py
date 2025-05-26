import pytest
from unittest.mock import patch, MagicMock, call, ANY
from pathlib import Path
import pandas as pd
import numpy as np # For OptimizedDataAccess mock attributes
import time # For checkpoint saving in OptimizedIndicesFinder

from lensing_ssc.core.preprocessing.indices import OptimizedIndicesFinder, IndicesValidator
from lensing_ssc.core.preprocessing.config import ProcessingConfig
from lensing_ssc.core.preprocessing.data_access import OptimizedDataAccess
from lensing_ssc.core.preprocessing.utils import CheckpointManager

# --- Fixtures for OptimizedIndicesFinder ---
@pytest.fixture
def mock_oif_config():
    # A more complete ProcessingConfig for OIF
    return ProcessingConfig(
        sheet_range=(0, 3), 
        extra_index=10,
        enable_progress_bar=False, 
        checkpoint_interval=1,
        chunk_size=5000, # from ProcessingConfig defaults
        cache_size_mb=1024,
        mmap_threshold=1000000,
        overwrite=False,
        num_workers=None,
        batch_size=10,
        log_level="INFO",
        cleanup_interval=50,
        max_cache_entries=1000,
        validate_input=True,
        strict_validation=False
    )

@pytest.fixture
def mock_oif_datadir(tmp_path):
    datadir = tmp_path / "oif_data"
    datadir.mkdir()
    return datadir

@pytest.fixture
def mock_oif_data_access(mock_oif_config): # Pass config to ODA mock if it uses it
    mock_oda = MagicMock(spec=OptimizedDataAccess)
    mock_oda.seed = 123
    mock_oda.is_sheet_empty.return_value = False
    mock_oda.get_sheet_bounds.side_effect = lambda sheet: (sheet * 100, (sheet + 1) * 100 -1) 
    mock_oda.find_aemit_change_point.return_value = None 
    mock_oda.monitor = MagicMock() 
    # If ODA constructor is called with config, mock_oif_config should be passed here or ensure ODA mock handles it.
    # The actual ODA class is instantiated inside OIF, so this mock_oif_data_access is for the *return value* of that.
    return mock_oda

@pytest.fixture
def mock_oif_checkpoint_manager():
    mock_cm = MagicMock(spec=CheckpointManager)
    mock_cm.load_checkpoint.return_value = {} 
    return mock_cm

@pytest.fixture
@patch("lensing_ssc.core.preprocessing.indices.OptimizedDataAccess")
@patch("lensing_ssc.core.preprocessing.indices.CheckpointManager")
def optimized_indices_finder_instance(
    MockCheckpointManager, MockOptimizedDataAccess,
    mock_oif_datadir, mock_oif_config, 
    mock_oif_data_access, 
    mock_oif_checkpoint_manager 
):
    MockOptimizedDataAccess.return_value = mock_oif_data_access 
    MockCheckpointManager.return_value = mock_oif_checkpoint_manager
    
    finder = OptimizedIndicesFinder(datadir=mock_oif_datadir, config=mock_oif_config)
    finder._mocks = {
        'data_access': mock_oif_data_access,
        'checkpoint_manager': mock_oif_checkpoint_manager,
        'OptimizedDataAccess_cls': MockOptimizedDataAccess, 
        'CheckpointManager_cls': MockCheckpointManager    
    }
    return finder

# --- Tests for OptimizedIndicesFinder --- 
def test_oif_init(optimized_indices_finder_instance, mock_oif_datadir, mock_oif_config):
    finder = optimized_indices_finder_instance
    assert finder.datadir == mock_oif_datadir
    assert finder.config == mock_oif_config
    assert finder.data == finder._mocks['data_access']
    assert finder.save_path == mock_oif_datadir / f"preproc_s{finder.data.seed}_indices.csv"
    assert finder.checkpoint_manager == finder._mocks['checkpoint_manager']
    assert finder.checkpoint_key == f"indices_s{finder.data.seed}"
    finder._mocks['OptimizedDataAccess_cls'].assert_called_once_with(mock_oif_datadir, mock_oif_config)
    finder._mocks['CheckpointManager_cls'].assert_called_once_with(mock_oif_datadir)

@patch("pandas.DataFrame.to_csv") 
@patch("pathlib.Path.exists", return_value=False) 
@patch("pandas.read_csv") 
def test_oif_find_indices_no_checkpoint_no_existing_csv(
    mock_pd_read_csv, mock_path_exists, mock_to_csv, optimized_indices_finder_instance
):
    finder = optimized_indices_finder_instance
    finder.config.sheet_range = (0, 2) 
    finder.data.is_sheet_empty.return_value = False
    finder._find_optimized_index = MagicMock(side_effect=lambda sheet, prev_end: (sheet*10, sheet*10 + 9))

    finder.find_indices()

    finder._mocks['checkpoint_manager'].load_checkpoint.assert_called_once()
    assert finder.data.is_sheet_empty.call_count == 2 
    mock_path_exists.assert_called_once_with(finder.save_path)
    mock_pd_read_csv.assert_not_called()
    assert finder._find_optimized_index.call_count == 2
    finder._find_optimized_index.assert_any_call(0, None)
    finder._find_optimized_index.assert_any_call(1, 9) 
    assert finder._mocks['checkpoint_manager'].save_checkpoint.call_count == 2
    mock_to_csv.assert_called_once() 
    df_saved = mock_to_csv.call_args[0][0]
    assert len(df_saved) == 2
    assert df_saved.iloc[0]['sheet'] == 0 and df_saved.iloc[0]['start'] == 0 and df_saved.iloc[0]['end'] == 9
    assert df_saved.iloc[1]['sheet'] == 1 and df_saved.iloc[1]['start'] == 10 and df_saved.iloc[1]['end'] == 19
    finder._mocks['checkpoint_manager'].clear_checkpoint.assert_called_once() 

@patch("pandas.DataFrame.to_csv")
@patch("pathlib.Path.exists", return_value=True)
@patch("pandas.read_csv")
def test_oif_find_indices_with_existing_csv(
    mock_pd_read_csv, mock_path_exists, mock_to_csv, optimized_indices_finder_instance
):
    finder = optimized_indices_finder_instance
    finder.config.sheet_range = (0, 3) 
    existing_df = pd.DataFrame([{'sheet': 0, 'start': 0, 'end': 9}])
    mock_pd_read_csv.return_value = existing_df
    finder._find_optimized_index = MagicMock(side_effect=lambda sheet, prev_end: (sheet*10, sheet*10 + 9))

    finder.find_indices()
    mock_path_exists.assert_called_with(finder.save_path)
    mock_pd_read_csv.assert_called_once_with(finder.save_path)
    assert finder._find_optimized_index.call_count == 2
    finder._find_optimized_index.assert_any_call(1, 9)
    finder._find_optimized_index.assert_any_call(2, 19)
    mock_to_csv.assert_called_once()
    df_saved = mock_to_csv.call_args[0][0]
    assert len(df_saved) == 3
    assert df_saved.iloc[0]['sheet'] == 0 
    assert df_saved.iloc[1]['sheet'] == 1 
    assert df_saved.iloc[2]['sheet'] == 2 

def test_oif_find_optimized_index_no_extra(optimized_indices_finder_instance):
    finder = optimized_indices_finder_instance
    finder.config.extra_index = 0 
    sheet_id = 1
    expected_start, expected_end = (100, 199)
    finder.data.get_sheet_bounds.return_value = (expected_start, expected_end)
    start, end = finder._find_optimized_index(sheet_id, prev_end=None)
    assert start == expected_start
    assert end == expected_end
    finder.data.get_sheet_bounds.assert_called_once_with(sheet_id)
    finder.data.find_aemit_change_point.assert_not_called()

def test_oif_find_optimized_index_with_extra_and_change_points(optimized_indices_finder_instance):
    finder = optimized_indices_finder_instance
    finder.config.extra_index = 20
    sheet_id = 1
    original_start, original_end = (100, 199)
    optimized_start_cp = 110
    optimized_end_cp = 189
    finder.data.get_sheet_bounds.return_value = (original_start, original_end)
    def find_cp_side_effect(idx, search_range, forward):
        if forward and idx == original_start: return optimized_start_cp
        if not forward and idx == original_end: return optimized_end_cp
        return None
    finder.data.find_aemit_change_point.side_effect = find_cp_side_effect
    start, end = finder._find_optimized_index(sheet_id, prev_end=None)
    assert start == optimized_start_cp
    assert end == optimized_end_cp
    assert finder.data.find_aemit_change_point.call_count == 2
    finder.data.find_aemit_change_point.assert_any_call(original_start, 20, forward=True)
    finder.data.find_aemit_change_point.assert_any_call(original_end, 20, forward=False)

# --- Fixtures for IndicesValidator ---
@pytest.fixture
def mock_iv_datadir(tmp_path):
    iv_dir = tmp_path / "iv_data"
    iv_dir.mkdir(exist_ok=True)
    return iv_dir

@pytest.fixture
def mock_iv_data_access():
    mock_oda = MagicMock(spec=OptimizedDataAccess)
    mock_oda.is_sheet_empty.return_value = False
    mock_oda.get_sheet_bounds.side_effect = lambda sheet_idx: (sheet_idx * 100, (sheet_idx + 1) * 100 - 1)
    return mock_oda

@pytest.fixture
def indices_validator_instance(mock_iv_datadir, mock_iv_data_access):
    return IndicesValidator(datadir=mock_iv_datadir, data_access=mock_iv_data_access)

@pytest.fixture
def valid_indices_file_for_validator(mock_iv_datadir):
    file_path = mock_iv_datadir / "valid_indices_val.csv"
    df = pd.DataFrame({'sheet': [0, 1], 'start': [0, 100], 'end': [99, 199]})
    df.to_csv(file_path, index=False)
    return file_path

# --- Tests for IndicesValidator ---
def test_iv_validate_indices_file_success(indices_validator_instance, valid_indices_file_for_validator):
    assert indices_validator_instance.validate_indices_file(valid_indices_file_for_validator) is True

def test_iv_validate_indices_file_not_found(indices_validator_instance, mock_iv_datadir):
    assert indices_validator_instance.validate_indices_file(mock_iv_datadir / "non_existent.csv") is False

def test_iv_validate_indices_file_bad_format(indices_validator_instance, mock_iv_datadir):
    bad_file = mock_iv_datadir / "bad_format.csv"
    bad_file.write_text("this,isnot,csv\n1,2,3,4,5")
    assert indices_validator_instance.validate_indices_file(bad_file) is False

def test_iv_validate_indices_file_missing_cols(indices_validator_instance, mock_iv_datadir):
    file_path = mock_iv_datadir / "missing_cols_val.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [0]})
    df.to_csv(file_path, index=False)
    assert indices_validator_instance.validate_indices_file(file_path) is False

def test_iv_validate_indices_file_start_gt_end(indices_validator_instance, mock_iv_datadir):
    file_path = mock_iv_datadir / "start_gt_end.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [100], 'end': [0]})
    df.to_csv(file_path, index=False)
    assert indices_validator_instance.validate_indices_file(file_path) is False

def test_iv_validate_indices_file_out_of_bounds(indices_validator_instance, mock_iv_datadir):
    file_path = mock_iv_datadir / "out_of_bounds.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [0], 'end': [150]})
    df.to_csv(file_path, index=False)
    assert indices_validator_instance.validate_indices_file(file_path) is False

def test_iv_detect_gaps_and_overlaps(indices_validator_instance, mock_iv_datadir):
    file_path = mock_iv_datadir / "gaps_overlaps.csv"
    df_gaps_overlaps = pd.DataFrame({
        'sheet': [0, 1, 2, 3, 4],
        'start': [0,  100, 170, 300, 330],
        'end':   [99, 180, 250, 290, 340] # Gap between 2 (end 250) and 3 (start 300)
                                          # Overlap between 1 (end 180) and 2 (start 170)
                                          # Overlap between 3 (end 290) and 4 (start 330) is wrong, should be end 350 start 330
    })
    # Corrected for overlaps and gaps based on standard definition (sorted by start then end)
    # df_gaps_overlaps = pd.DataFrame({
    #     'sheet': [0, 1, 2, 3, 4],
    #     'start': [0,  100, 170, 300, 330],
    #     'end':   [99, 180, 250, 350, 340] 
    # })
    # Test data: s0(0,99), s1(100,180), s2(170,250), s3(300,350), s4(330,360)
    # Overlap: s1 and s2 (170 < 180)
    # Gap: s2 and s3 (250 < 300)
    # Overlap: s3 and s4 (330 < 350)
    df_gaps_overlaps = pd.DataFrame({
        'sheet' : pd.Series([0, 1, 2, 3, 4], dtype=int),
        'start' : pd.Series([0, 100, 170, 300, 330], dtype=int),
        'end'   : pd.Series([99, 180, 250, 350, 360], dtype=int)
    })

    df_gaps_overlaps.to_csv(file_path, index=False)
    results = indices_validator_instance.detect_gaps_and_overlaps(file_path)
    
    assert len(results['gaps']) == 1
    assert any(g['prev_sheet'] == 2 and g['next_sheet'] == 3 and g['gap_size'] == (300 - 250 -1) for g in results['gaps'])

    assert len(results['overlaps']) == 2
    assert any(o['prev_sheet'] == 1 and o['next_sheet'] == 2 and o['overlap_size'] == (180 - 170 + 1) for o in results['overlaps'])
    assert any(o['prev_sheet'] == 3 and o['next_sheet'] == 4 and o['overlap_size'] == (350 - 330 + 1) for o in results['overlaps']) 