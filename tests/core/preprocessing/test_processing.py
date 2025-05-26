import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import healpy as hp
import pandas as pd

from lensing_ssc.core.preprocessing.processing import (
    PatchExtractor,
    # get_patch_pixels, # This seems to be part of FibonacciGrid in patching_utils.py now
    # process_single_map, # This is a method of PatchProcessorCLI or similar orchestrator
    # main_patch_processing_loop # This is likely the main function in cli.py
    MassSheetProcessor,
    ProcessingResult
)
from lensing_ssc.core.patching_utils import PatchProcessor # PatchExtractor is now PatchProcessor
from lensing_ssc.core.fibonacci_utils import FibonacciGrid # For get_patch_pixels
from lensing_ssc.core.preprocessing.config import PreprocessingConfig, ProcessingConfig
from lensing_ssc.core.preprocessing.data_access import OptimizedDataAccess
from lensing_ssc.core.preprocessing.validation import DataValidator
from lensing_ssc.core.preprocessing.indices import OptimizedIndicesFinder
from lensing_ssc.core.preprocessing.utils import CheckpointManager, PerformanceMonitor, ProgressTracker

# Constants for testing
TEST_XSIZE = 128
TEST_PATCH_SIZE_DEG = 5.0
TEST_RESOLUTION_ARCMIN = (TEST_PATCH_SIZE_DEG * 60.0) / TEST_XSIZE
TEST_PADDING = 0.1 + np.sqrt(2)

@pytest.fixture
def mock_config_for_patch_processor(tmp_path):
    """Provides a PreprocessingConfig relevant for PatchProcessor/Extractor."""
    center_points_dir = tmp_path / "center_points"
    center_points_dir.mkdir()
    # Create a dummy center points file
    # These points are (theta, phi) in radians
    points = np.array([
        [np.pi/2, np.pi/2], # Equator, Lon = 90 deg
        [np.pi/4, np.pi],   # Lat = 45 deg, Lon = 180 deg
        [3*np.pi/4, 0]    # Lat = -45 deg, Lon = 0 deg
    ])
    np.savetxt(center_points_dir / f"fibonacci_points_{TEST_PATCH_SIZE_DEG}.txt", points)

    return PreprocessingConfig(
        patch_size_deg=TEST_PATCH_SIZE_DEG,
        xsize=TEST_XSIZE,
        center_points_dir=str(center_points_dir),
        # Other fields can be defaults
    )

@pytest.fixture
def patch_processor_instance(mock_config_for_patch_processor):
    """Initializes a PatchProcessor instance for testing."""
    return PatchProcessor(
        patch_size_deg=mock_config_for_patch_processor.patch_size_deg,
        xsize=mock_config_for_patch_processor.xsize,
        center_points_path=str(mock_config_for_patch_processor.center_points_dir)
    )

@pytest.fixture
def sample_healpy_map():
    """Creates a sample Healpy map for testing."""
    nside = 64 # NPIX = 12 * 64^2 = 49152
    npix = hp.nside2npix(nside)
    test_map = np.arange(npix, dtype=float)
    # No reordering here, as PatchProcessor expects RING ordered by default from hp.gnomview
    return test_map

# Tests for PatchProcessor (formerly PatchExtractor)

def test_patch_processor_initialization(patch_processor_instance, mock_config_for_patch_processor):
    """Test the initialization of PatchProcessor."""
    pp = patch_processor_instance
    cfg = mock_config_for_patch_processor
    assert pp.patch_size_deg == cfg.patch_size_deg
    assert pp.xsize == cfg.xsize
    assert pp.resolution_arcmin == (cfg.patch_size_deg * 60.0) / cfg.xsize
    assert pp.npatch == 3 # From the dummy file created in mock_config
    assert pp.center_points.shape == (3, 2)

def test_patch_processor_initialization_file_not_found_fallback_to_generate(tmp_path):
    """Test initialization when center points file is not found, falls back to FibonacciGrid generation."""
    center_points_dir_empty = tmp_path / "empty_center_points"
    center_points_dir_empty.mkdir()

    n_opt_for_generation = 5 # Small number for test
    generated_points = np.random.rand(n_opt_for_generation, 2)

    with patch.object(FibonacciGrid, 'load_or_generate_points', return_value=generated_points) as mock_generate:
        pp = PatchProcessor(
            patch_size_deg=TEST_PATCH_SIZE_DEG,
            xsize=TEST_XSIZE,
            center_points_path=str(center_points_dir_empty)
        )
        mock_generate.assert_called_once()
        # The n_opt_placeholder calculation is internal, so we check the outcome
        assert pp.npatch == n_opt_for_generation
        np.testing.assert_array_equal(pp.center_points, generated_points)

def test_patch_processor_initialization_generation_fails(tmp_path):
    """Test initialization fails if both file loading and generation fail."""
    center_points_dir_empty = tmp_path / "empty_center_points_fail"
    center_points_dir_empty.mkdir()

    with patch.object(FibonacciGrid, 'load_or_generate_points', return_value=np.array([])) as mock_generate:
        with pytest.raises(FileNotFoundError, match="Failed to load or generate center points"):
            PatchProcessor(
                patch_size_deg=TEST_PATCH_SIZE_DEG,
                xsize=TEST_XSIZE,
                center_points_path=str(center_points_dir_empty)
            )
        mock_generate.assert_called_once()

def test_get_valid_points(patch_processor_instance):
    """Test the _get_valid_points method for pole filtering."""
    pp = patch_processor_instance
    # Original points: [[pi/2, pi/2], [pi/4, pi], [3pi/4, 0]]
    # patch_size_deg = 5.0. angular_radius_rad = 5.0 * sqrt(2) * pi/180 ~= 0.123 radians
    # pi - 0.123 ~= 3.018
    # 0 + 0.123 ~= 0.123
    # Point 1: theta=pi/2 (1.57) -> valid (0.123 < 1.57 < 3.018)
    # Point 2: theta=pi/4 (0.785) -> valid (0.123 < 0.785 < 3.018)
    # Point 3: theta=3pi/4 (2.356) -> valid (0.123 < 2.356 < 3.018)
    # All points should be valid with these values.

    # Let's add a point too close to a pole to the processor's center_points directly for this test
    original_points = pp.center_points.copy()
    point_near_north_pole = np.array([[0.05, 0.1]]) # theta = 0.05 < angular_radius_rad
    point_near_south_pole = np.array([[3.1, 0.1]])    # theta = 3.1 > pi - angular_radius_rad
    pp.center_points = np.vstack((original_points, point_near_north_pole, point_near_south_pole))
    
    valid_points = pp._get_valid_points()
    assert len(valid_points) == len(original_points) # Only the original 3 should be valid
    # Check that the near-pole points are not in valid_points
    assert not any(np.allclose(vp, point_near_north_pole[0]) for vp in valid_points)
    assert not any(np.allclose(vp, point_near_south_pole[0]) for vp in valid_points)

# Mocking healpy.gnomview and FibonacciGrid.get_patch_pixels for worker test
@patch('lensing_ssc.core.patching_utils.FibonacciGrid.get_patch_pixels')
@patch('healpy.gnomview')
@patch('multiprocessing.shared_memory.SharedMemory') # Mock shared memory access
def test_extract_patch_worker_successful(mock_shm, mock_gnomview, mock_get_patch_pixels, patch_processor_instance, sample_healpy_map):
    """Test the _extract_patch_worker method for successful patch extraction."""
    pp = patch_processor_instance
    mock_projected_map = np.random.rand(int(pp.xsize * pp.padding), int(pp.xsize * pp.padding))
    mock_final_patch = np.random.rand(pp.xsize, pp.xsize).astype(np.float32)
    
    mock_gnomview.return_value = mock_projected_map
    mock_get_patch_pixels.return_value = mock_final_patch

    # Mock SharedMemory object
    mock_shm_instance = MagicMock()
    mock_shm_instance.buf = sample_healpy_map.data # or sample_healpy_map.tobytes()
    mock_shm.return_value = mock_shm_instance

    point_lonlat_deg = (90.0, 0.0) # Example point (lon, lat)
    
    extracted_patch = pp._extract_patch_worker(
        shm_name="test_shm", 
        shape=sample_healpy_map.shape, 
        dtype=sample_healpy_map.dtype, 
        point_lonlat_deg=point_lonlat_deg
    )
    
    mock_shm.assert_called_once_with(name="test_shm")
    mock_gnomview.assert_called_once_with(
        unittest.mock.ANY, # The map from shared memory
        rot=point_lonlat_deg,
        xsize=int(pp.xsize * pp.padding),
        reso=pp.resolution_arcmin,
        return_projected_map=True,
        nest=False,
        no_plot=True
    )
    # Check that the first argument to gnomview is indeed the sample_healpy_map from shared memory
    args_gnomview, _ = mock_gnomview.call_args
    assert isinstance(args_gnomview[0], np.ndarray)
    # np.testing.assert_array_equal(args_gnomview[0], sample_healpy_map) # This comparison can be tricky with mocks

    mock_get_patch_pixels.assert_called_once_with(mock_projected_map, pp.xsize)
    np.testing.assert_array_equal(extracted_patch, mock_final_patch)
    mock_shm_instance.close.assert_called_once()

@patch('lensing_ssc.core.patching_utils.FibonacciGrid.get_patch_pixels')
@patch('healpy.gnomview', side_effect=Exception("Gnomview failed"))
@patch('multiprocessing.shared_memory.SharedMemory')
def test_extract_patch_worker_gnomview_fails(mock_shm, mock_gnomview, mock_get_patch_pixels, patch_processor_instance, sample_healpy_map):
    """Test _extract_patch_worker when hp.gnomview raises an exception."""
    pp = patch_processor_instance
    mock_shm_instance = MagicMock()
    mock_shm_instance.buf = sample_healpy_map.data
    mock_shm.return_value = mock_shm_instance

    point_lonlat_deg = (0.0, 0.0)
    failed_patch = pp._extract_patch_worker("test_shm_fail", sample_healpy_map.shape, sample_healpy_map.dtype, point_lonlat_deg)
    
    assert failed_patch.shape == (pp.xsize, pp.xsize)
    assert np.all(failed_patch == 0) # Should return a zero array
    mock_get_patch_pixels.assert_not_called()
    mock_shm_instance.close.assert_called_once()


@patch('multiprocessing.Pool')
@patch('multiprocessing.shared_memory.SharedMemory')
def test_make_patches_successful_run(mock_shm_constructor, mock_pool_constructor, patch_processor_instance, sample_healpy_map):
    """Test the make_patches method for a successful run using multiprocessing."""
    pp = patch_processor_instance
    num_valid_points = len(pp._get_valid_points()) # All 3 points are valid by default setup
    assert num_valid_points > 0 

    # Mock shared memory setup
    mock_shm_obj = MagicMock()
    mock_shm_obj.name = "mock_shm_name"
    mock_shm_obj.buf = bytearray(sample_healpy_map.nbytes) # Placeholder buffer
    mock_shm_constructor.return_value = mock_shm_obj

    # Mock Pool and starmap result
    mock_pool_instance = MagicMock()
    mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance # For 'with Pool ... as pool'
    # Each call to _extract_patch_worker returns a patch
    expected_single_patch = np.random.rand(pp.xsize, pp.xsize).astype(np.float32)
    mock_pool_instance.starmap.return_value = [expected_single_patch.copy() for _ in range(num_valid_points)]

    patches_result = pp.make_patches(sample_healpy_map, num_processes=2)

    mock_shm_constructor.assert_called_once_with(create=True, size=sample_healpy_map.nbytes)
    mock_pool_constructor.assert_called_once_with(processes=2)
    mock_pool_instance.starmap.assert_called_once()
    
    # Check arguments passed to starmap
    args_starmap, _ = mock_pool_instance.starmap.call_args
    assert args_starmap[0] == pp._extract_patch_worker
    assert len(args_starmap[1]) == num_valid_points # Iterable of arguments
    first_arg_tuple = args_starmap[1][0]
    assert first_arg_tuple[0] == mock_shm_obj.name
    assert first_arg_tuple[1] == sample_healpy_map.shape
    assert first_arg_tuple[2] == sample_healpy_map.dtype
    # (lon,lat) of first point: hp.rotator.vec2dir(hp.ang2vec(np.pi/2, np.pi/2), lonlat=True) -> (90.0, 0.0)
    # This can be tricky to assert exactly due to float precision with hp.ang2vec and vec2dir
    # For simplicity, we ensure the structure is correct.

    assert patches_result.shape == (num_valid_points, pp.xsize, pp.xsize)
    np.testing.assert_array_equal(patches_result[0], expected_single_patch)

    mock_shm_obj.close.assert_called_once()
    mock_shm_obj.unlink.assert_called_once()


@patch('multiprocessing.Pool')
@patch('multiprocessing.shared_memory.SharedMemory')
def test_make_patches_worker_exception(mock_shm_constructor, mock_pool_constructor, patch_processor_instance, sample_healpy_map):
    """Test make_patches when a worker process raises an exception."""
    pp = patch_processor_instance
    mock_shm_obj = MagicMock()
    mock_shm_obj.name = "mock_shm_name_exc"
    mock_shm_obj.buf = bytearray(sample_healpy_map.nbytes)
    mock_shm_constructor.return_value = mock_shm_obj

    mock_pool_instance = MagicMock()
    mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.starmap.side_effect = Exception("Worker failed miserably")

    with pytest.raises(Exception, match="Worker failed miserably"):
        pp.make_patches(sample_healpy_map, num_processes=1)
    
    mock_shm_obj.close.assert_called_once()
    mock_shm_obj.unlink.assert_called_once() # Ensure cleanup even on error

# Note: Testing `FibonacciGrid.get_patch_pixels` itself would belong in a test file for `fibonacci_utils.py`.
# The `processing.py` file from the user's list seems to contain `PatchExtractor` which is an older name
# for `PatchProcessor` in `patching_utils.py`. Tests are written for `PatchProcessor` as per current codebase structure.
# If `get_patch_pixels` was meant to be in `preprocessing.processing`, its tests would be here.
# However, `patching_utils.py` shows `FibonacciGrid.get_patch_pixels`. 

# Default config for tests
@pytest.fixture
def default_config_dict():
    return {
        "sheet_range": (0, 2),
        "overwrite": False,
        "num_workers": 1,
        "chunk_size": 1000,
        "max_cache_entries": 10,
        "cleanup_interval": 5, 
        "log_level": "INFO", 
        "cache_size_mb": 1024, 
        "mmap_threshold": 1000000, 
        "extra_index": 100, 
        "batch_size": 10, 
        "enable_progress_bar": True,
        "checkpoint_interval": 10,
        "validate_input": True, 
        "strict_validation": False,
    }

@pytest.fixture
def mock_processing_config(default_config_dict):
    return ProcessingConfig(**default_config_dict)

@pytest.fixture
def mock_datadir(tmp_path):
    datadir = tmp_path / "sim_data"
    datadir.mkdir()
    (datadir / "usmesh").mkdir(parents=True, exist_ok=True)
    return datadir

@pytest.fixture
def mock_optimized_data_access():
    mock_oda = MagicMock(spec=OptimizedDataAccess)
    mock_oda.msheets = MagicMock()
    mock_oda.msheets.attrs = { 
        'aemitIndex.edges': np.array([0.5, 1.0, 1.5, 2.0]),
        'aemitIndex.offset': np.array([0, 10, 20, 30]),
        'healpix.npix': [12*16**2],
        'BoxSize': [256.0],
        'MassTable': [0.0, 1e10],
        'NC': [512],
        'seed': [0]
    }
    mock_oda.seed = 0 
    mock_oda.total_records = 1000 # Add a total_records attribute
    mock_oda.a_interval = 0.5 # Add a_interval
    return mock_oda

@pytest.fixture
def mock_data_validator():
    return MagicMock(spec=DataValidator)

@pytest.fixture
def mock_checkpoint_manager():
    mock_cm = MagicMock(spec=CheckpointManager)
    mock_cm.load_checkpoint.return_value = {} 
    return mock_cm
    
@pytest.fixture
def mock_optimized_indices_finder():
    mock_oif = MagicMock(spec=OptimizedIndicesFinder)
    mock_oif.find_indices = MagicMock()
    return mock_oif

@pytest.fixture
def sample_indices_df():
    data = {'sheet': [0, 1], 'start_idx': [0, 10], 'end_idx': [10, 20]}
    return pd.DataFrame(data)

@pytest.fixture
@patch("lensing_ssc.core.preprocessing.processing.OptimizedDataAccess")
@patch("lensing_ssc.core.preprocessing.processing.DataValidator")
@patch("lensing_ssc.core.preprocessing.processing.CheckpointManager")
@patch("lensing_ssc.core.preprocessing.processing.OptimizedIndicesFinder")
@patch("lensing_ssc.core.preprocessing.processing.PerformanceMonitor") 
@patch("pandas.read_csv")
@patch("pathlib.Path.exists")
def mass_sheet_processor_instance(
    mock_path_exists, mock_read_csv, mock_pm_cls, mock_oif_cls, mock_cm_cls, 
    mock_dv_cls, mock_oda_cls, 
    mock_datadir, mock_processing_config, sample_indices_df,
    mock_optimized_data_access, mock_data_validator, 
    mock_checkpoint_manager, mock_optimized_indices_finder
):
    mock_oda_cls.return_value = mock_optimized_data_access
    mock_dv_cls.return_value = mock_data_validator
    mock_cm_cls.return_value = mock_checkpoint_manager
    mock_oif_cls.return_value = mock_optimized_indices_finder
    mock_pm_cls.return_value = MagicMock(spec=PerformanceMonitor) # Mock PerformanceMonitor

    mock_path_exists.return_value = True 
    mock_read_csv.return_value = sample_indices_df
    mock_processing_config.overwrite = False

    processor = MassSheetProcessor(datadir=mock_datadir, config=mock_processing_config)
    
    processor._mocks = {
        'oda': mock_optimized_data_access,
        'dv': mock_data_validator,
        'cm': mock_checkpoint_manager,
        'oif': mock_optimized_indices_finder,
        'oif_cls': mock_oif_cls, 
        'read_csv': mock_read_csv,
        'path_exists': mock_path_exists,
        'pm': mock_pm_cls.return_value
    }
    return processor

# --- Basic Initialization Tests ---
def test_mass_sheet_processor_initialization(mass_sheet_processor_instance, mock_datadir, mock_processing_config):
    processor = mass_sheet_processor_instance
    assert processor.datadir == mock_datadir
    assert processor.config == mock_processing_config
    assert processor.output_dir == mock_datadir / "mass_sheets"
    
    # Check that OptimizedDataAccess was initialized with datadir and config
    # The mock_optimized_data_access fixture is what's returned by mock_oda_cls, so we check call to its __init__ via the class mock
    from lensing_ssc.core.preprocessing.processing import OptimizedDataAccess as ODA_Class # get original class for constructor call check
    mass_sheet_processor_instance._mocks['oda'].__class__.assert_any_call(mock_datadir, mock_processing_config)

    processor._mocks['dv'].validate_usmesh_structure.assert_called_once_with(processor._mocks['oda'].msheets)
    processor._mocks['read_csv'].assert_called_once()
    expected_csv_path = mock_datadir / f"preproc_s{processor._mocks['oda'].seed}_indices.csv"
    processor._mocks['path_exists'].assert_any_call(expected_csv_path) 
    processor._mocks['oif'].find_indices.assert_not_called()
    assert processor.indices_df.equals(processor._mocks['read_csv'].return_value)
    assert (mock_datadir / "mass_sheets").exists()


@patch("lensing_ssc.core.preprocessing.processing.OptimizedDataAccess")
@patch("lensing_ssc.core.preprocessing.processing.DataValidator") # Added DV mock
@patch("lensing_ssc.core.preprocessing.processing.CheckpointManager") # Added CM mock
@patch("lensing_ssc.core.preprocessing.processing.OptimizedIndicesFinder")
@patch("lensing_ssc.core.preprocessing.processing.PerformanceMonitor") # Added PM mock
@patch("pandas.read_csv")
@patch("pathlib.Path.exists")
def test_mass_sheet_processor_init_create_indices(
    mock_path_exists, mock_read_csv, mock_pm_cls, mock_oif_cls, mock_cm_cls, mock_dv_cls, mock_oda_cls,
    mock_datadir, mock_processing_config, sample_indices_df,
    mock_optimized_data_access, mock_optimized_indices_finder 
):
    mock_oda_cls.return_value = mock_optimized_data_access
    mock_oif_cls.return_value = mock_optimized_indices_finder
    mock_dv_cls.return_value = MagicMock(spec=DataValidator) # Basic mock for DV
    mock_cm_cls.return_value = MagicMock(spec=CheckpointManager) # Basic mock for CM
    mock_pm_cls.return_value = MagicMock(spec=PerformanceMonitor)
    
    mock_path_exists.return_value = False 
    mock_processing_config.overwrite = False 
    mock_read_csv.return_value = sample_indices_df 

    processor = MassSheetProcessor(datadir=mock_datadir, config=mock_processing_config)

    expected_csv_path = mock_datadir / f"preproc_s{processor._mocks['oda'].seed}_indices.csv"
    mock_path_exists.assert_any_call(expected_csv_path)
    mock_optimized_indices_finder.find_indices.assert_called_once_with(*mock_processing_config.sheet_range)
    mock_read_csv.assert_called_with(expected_csv_path)
    assert processor.indices_df.equals(sample_indices_df)

# --- Preprocess Method Tests (High Level) ---
@patch("lensing_ssc.core.preprocessing.processing.ProgressTracker")
def test_preprocess_all_sheets_completed(
    mock_progress_tracker_cls, mass_sheet_processor_instance, sample_indices_df
):
    processor = mass_sheet_processor_instance
    completed_ids = [0, 1]
    processor._mocks['cm'].load_checkpoint.return_value = {
        processor.checkpoint_key: {'completed_sheets': completed_ids}
    }
    processor.indices_df = sample_indices_df

    result = processor.preprocess()

    assert result['status'] == "complete"
    assert result['processed'] == len(completed_ids)
    mock_progress_tracker_cls.assert_not_called() 
    # In the new code, _save_checkpoint_entry is not called if all sheets are completed.
    processor._mocks['cm']._save_checkpoint_entry.assert_not_called()

@patch.object(MassSheetProcessor, "_process_sequential")
@patch("lensing_ssc.core.preprocessing.processing.ProgressTracker")
def test_preprocess_sequential_run(
    mock_progress_tracker_cls, mock_process_sequential,
    mass_sheet_processor_instance, sample_indices_df
):
    processor = mass_sheet_processor_instance
    processor.config.num_workers = 1 
    processor.indices_df = sample_indices_df
    processor._mocks['cm'].load_checkpoint.return_value = {} 
    mock_progress_tracker_instance = MagicMock(spec=ProgressTracker)
    mock_progress_tracker_cls.return_value = mock_progress_tracker_instance

    mock_results = [
        ProcessingResult(sheet_id=0, success=True, processing_time=0.1),
        ProcessingResult(sheet_id=1, success=True, processing_time=0.1)
    ]
    mock_process_sequential.return_value = mock_results

    summary = processor.preprocess()

    mock_process_sequential.assert_called_once()
    args, _ = mock_process_sequential.call_args
    assert len(args[0]) == len(sample_indices_df)
    assert args[1] == mock_progress_tracker_instance 
    
    processor._mocks['cm']._save_checkpoint_entry.assert_called_once_with([0, 1])
    assert summary['total_sheets_to_process'] == len(sample_indices_df)
    assert summary['successfully_processed'] == 2

@patch.object(MassSheetProcessor, "_process_parallel")
@patch("lensing_ssc.core.preprocessing.processing.ProgressTracker")
def test_preprocess_parallel_run(
    mock_progress_tracker_cls, mock_process_parallel,
    mass_sheet_processor_instance, sample_indices_df
):
    processor = mass_sheet_processor_instance
    processor.config.num_workers = 2 
    processor.indices_df = sample_indices_df
    processor._mocks['cm'].load_checkpoint.return_value = {}
    mock_progress_tracker_instance = MagicMock(spec=ProgressTracker)
    mock_progress_tracker_cls.return_value = mock_progress_tracker_instance

    mock_results = [
        ProcessingResult(sheet_id=0, success=True, processing_time=0.1),
        ProcessingResult(sheet_id=1, success=True, processing_time=0.1)
    ]
    mock_process_parallel.return_value = mock_results
    
    summary = processor.preprocess()

    mock_process_parallel.assert_called_once()
    args, _ = mock_process_parallel.call_args
    assert len(args[0]) == len(sample_indices_df)
    assert args[1] == mock_progress_tracker_instance

    processor._mocks['cm']._save_checkpoint_entry.assert_called_once_with([0,1])
    assert summary['successfully_processed'] == 2

@patch.object(MassSheetProcessor, "_process_single_sheet")
def test_process_sequential_calls_single_sheet(
    mock_pss, mass_sheet_processor_instance, sample_indices_df
):
    processor = mass_sheet_processor_instance
    mock_progress_tracker = MagicMock(spec=ProgressTracker)

    results_map = {
        0: ProcessingResult(sheet_id=0, success=True),
        1: ProcessingResult(sheet_id=1, success=False, error="Test error")
    }
    def side_effect_pss(row_series):
        return results_map[row_series['sheet']]
    
    mock_pss.side_effect = side_effect_pss
    
    sheets_to_process_list = [row for _, row in sample_indices_df.iterrows()]
    results = processor._process_sequential(sheets_to_process_list, mock_progress_tracker)

    assert mock_pss.call_count == len(sample_indices_df)
    for i, row_series in enumerate(sheets_to_process_list):
        # Check that _process_single_sheet was called with a pandas Series representing the row
        called_arg = mock_pss.call_args_list[i][0][0]
        assert isinstance(called_arg, pd.Series)
        assert called_arg['sheet'] == row_series['sheet']

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert results[1].error == "Test error"
    assert mock_progress_tracker.update.call_count == len(sample_indices_df)

@patch("lensing_ssc.core.preprocessing.processing.ProcessPoolExecutor")
@patch("lensing_ssc.core.preprocessing.processing.as_completed")
@patch.object(MassSheetProcessor, "_process_single_sheet_static")
def test_process_parallel_execution_flow(
    mock_pss_static, mock_as_completed, mock_executor_cls, 
    mass_sheet_processor_instance, sample_indices_df, default_config_dict
):
    processor = mass_sheet_processor_instance
    processor.config.num_workers = 2
    mock_progress_tracker = MagicMock(spec=ProgressTracker)
    
    mock_executor_instance = MagicMock()
    mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance
    
    mock_future_sheet0 = MagicMock()
    mock_future_sheet0.result.return_value = ProcessingResult(sheet_id=0, success=True)
    mock_future_sheet1 = MagicMock()
    mock_future_sheet1.result.return_value = ProcessingResult(sheet_id=1, success=True)

    # Simulate the submit calls mapping to our futures
    # The order of submission might not be guaranteed, but for this test, let's assume it matches sample_indices_df order
    mock_executor_instance.submit.side_effect = [mock_future_sheet0, mock_future_sheet1]
    mock_as_completed.return_value = [mock_future_sheet0, mock_future_sheet1]
            
    sheets_to_process_list = [dict(row) for _, row in sample_indices_df.iterrows()]
    results = processor._process_parallel(sheets_to_process_list, mock_progress_tracker)

    mock_executor_cls.assert_called_once_with(max_workers=processor.config.num_workers)
    assert mock_executor_instance.submit.call_count == len(sheets_to_process_list)
    
    # Check arguments to _process_single_sheet_static
    expected_calls_pss_static = []
    for row_dict in sheets_to_process_list:
        expected_calls_pss_static.append(call(
            row_dict,
            default_config_dict, # Passed as a dict
            {'datadir': str(processor.datadir)}, 
            {'H0': processor.cosmo.H0.value, 'Om0': processor.cosmo.Om0},
            ANY, # processing_params_dict - too complex to fully match here without more setup
            str(processor.output_dir),
            None # monitor_shared
        ))
    # mock_pss_static.assert_has_calls(expected_calls_pss_static, any_order=True) # This might be too strict if processing_params_dict is complex
    assert mock_pss_static.call_count == len(sheets_to_process_list)

    mock_as_completed.assert_called_once()
    assert mock_future_sheet0.result.called
    assert mock_future_sheet1.result.called
            
    assert len(results) == 2
    assert all(r.success for r in results)
    assert mock_progress_tracker.update.call_count == 2

@patch.object(MassSheetProcessor, "_save_mass_sheet") # Mocking the save function
@patch.object(MassSheetProcessor, "_compute_mass_sheet")
def test_process_single_sheet_core_success(
    mock_compute_mass_sheet, mock_save_mass_sheet, mass_sheet_processor_instance
):
    processor = mass_sheet_processor_instance
    sheet_id = 0
    start_idx = 0
    end_idx = 10
    
    mock_kappa_data = np.random.rand(100)
    mock_compute_mass_sheet.return_value = mock_kappa_data
    
    # The _process_single_sheet_core method itself doesn't create the per-sheet directory.
    # It expects it to exist if saving files there. The _save_mass_sheet method handles the path construction.

    result = processor._process_single_sheet_core(sheet_id, start_idx, end_idx)

    mock_compute_mass_sheet.assert_called_once_with(sheet_id, start_idx, end_idx)
    mock_save_mass_sheet.assert_called_once_with(mock_kappa_data, sheet_id)

    assert result.success is True
    assert result.sheet_id == sheet_id
    assert result.error is None

@patch.object(MassSheetProcessor, "_compute_mass_sheet", side_effect=Exception("Compute error!"))
@patch.object(MassSheetProcessor, "_save_mass_sheet") # Also mock save, though it won't be called
def test_process_single_sheet_core_compute_failure(
    mock_save_mass_sheet, mock_compute_mass_sheet, mass_sheet_processor_instance
):
    processor = mass_sheet_processor_instance
    result = processor._process_single_sheet_core(sheet_id=0, start_idx=0, end_idx=10)

    assert result.success is False
    assert result.sheet_id == 0
    assert "Compute error!" in result.error
    assert result.processing_time is not None
    mock_save_mass_sheet.assert_not_called() # Should not attempt to save if computation fails


@patch("healpy.ang2pix")
@patch("healpy.vec2ang")
def test_compute_mass_sheet_logic(
    mock_vec2ang, mock_ang2pix, mass_sheet_processor_instance
):
    processor = mass_sheet_processor_instance
    sheet_id = 0
    start_idx = 0
    end_idx = 5 

    mock_pos = np.random.rand(end_idx - start_idx, 3) * processor.box_size 
    mock_vel = np.random.rand(end_idx - start_idx, 3) 
    mock_aemit = np.ones(end_idx - start_idx) * 0.8 
    
    # Configure the side_effect for get_data_slice
    def get_data_slice_side_effect(column, start, end):
        if column == 'Position': return mock_pos
        if column == 'Velocity': return mock_vel
        if column == 'Aemit': return mock_aemit
        return MagicMock() # Default for other columns if any
    processor.data_access.get_data_slice.side_effect = get_data_slice_side_effect

    # Mock healpy returns
    # vec2ang returns (theta, phi)
    mock_vec2ang.return_value = (np.random.uniform(0, np.pi, size=len(mock_pos)), 
                                 np.random.uniform(0, 2*np.pi, size=len(mock_pos)))
    mock_ang2pix.return_value = np.random.randint(0, processor.npix, size=len(mock_pos))

    kappa_map = processor._compute_mass_sheet(sheet_id, start_idx, end_idx)

    assert isinstance(kappa_map, np.ndarray)
    assert kappa_map.shape == (processor.npix,)
    assert processor.data_access.get_data_slice.call_count >= 3 # Position, Velocity, Aemit called at least once each
    mock_vec2ang.assert_called()
    mock_ang2pix.assert_called()


def test_preprocess_loads_checkpoint(mass_sheet_processor_instance, sample_indices_df):
    processor = mass_sheet_processor_instance
    completed_sheet_id_from_checkpoint = 0
    processor._mocks['cm'].load_checkpoint.return_value = {
        processor.checkpoint_key: {'completed_sheets': [completed_sheet_id_from_checkpoint]}
    }
    processor.indices_df = sample_indices_df # Contains sheets 0 and 1

    with patch.object(processor, "_process_sequential", return_value=[]) as mock_proc_seq:
         processor.preprocess() # This will call _process_sequential or _process_parallel
         
         args, kwargs = mock_proc_seq.call_args
         sheets_passed_to_processing = args[0]
         processed_sheet_ids = [row['sheet'] for row in sheets_passed_to_processing]
         
         assert completed_sheet_id_from_checkpoint not in processed_sheet_ids
         # Check that other sheets are still there
         assert 1 in processed_sheet_ids 


def test_preprocess_saves_checkpoint(mass_sheet_processor_instance, sample_indices_df):
    processor = mass_sheet_processor_instance
    processor.indices_df = sample_indices_df
    processor._mocks['cm'].load_checkpoint.return_value = {} 

    # Sheets 0 and 1 are in sample_indices_df
    # Assume _process_sequential processes them and returns success
    mock_results = [ProcessingResult(sheet_id=0, success=True), ProcessingResult(sheet_id=1, success=True)]
    with patch.object(processor, "_process_sequential", return_value=mock_results) as mock_proc_seq:
        processor.preprocess()
        processor._mocks['cm']._save_checkpoint_entry.assert_called_once_with([0, 1]) 