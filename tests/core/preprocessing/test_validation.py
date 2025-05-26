import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd
import healpy as hp # healpy is used by a method in DataValidator

from lensing_ssc.core.preprocessing.validation import DataValidator, ValidationError

@pytest.fixture
def data_validator_instance():
    """Provides an instance of DataValidator."""
    return DataValidator()

# --- Tests for __init__ ---
def test_data_validator_init(data_validator_instance):
    assert data_validator_instance.required_columns == ['ID', 'Mass', 'Aemit']
    assert data_validator_instance.required_attrs == [
        'aemitIndex.edges', 'aemitIndex.offset', 'healpix.npix',
        'BoxSize', 'MassTable', 'NC'
    ]

# --- Tests for validate_usmesh_structure and its sub-methods ---
@pytest.fixture
def mock_bigfile_catalog():
    catalog = MagicMock()
    # spec=BigFileCatalog might be too restrictive if BigFileCatalog is complex to mock fully
    catalog.columns = ['ID', 'Mass', 'Aemit', 'OtherCol']
    catalog.attrs = {
        'aemitIndex.edges': np.array([0.1, 0.2, 0.3]),
        'aemitIndex.offset': np.array([0, 100, 200, 300]),
        'healpix.npix': 12 * 16**2,
        'BoxSize': 256.0,
        'MassTable': [0.0, 1e10],
        'NC': 512,
        'extra_attr': 'some_value'
    }
    catalog.size = 300
    mock_aemit_data = np.random.uniform(0.01, 0.99, size=100)
    # Mocking behavior for catalog['Aemit'][:sample_size].compute()
    mock_column_slice = MagicMock()
    mock_column_slice.compute.return_value = mock_aemit_data
    catalog.__getitem__.return_value = mock_column_slice # For msheets['Aemit']
    return catalog

def test_validate_usmesh_structure_success(data_validator_instance, mock_bigfile_catalog):
    assert data_validator_instance.validate_usmesh_structure(mock_bigfile_catalog) is True

def test_validate_columns_missing(data_validator_instance, mock_bigfile_catalog):
    mock_bigfile_catalog.columns = ['ID', 'Mass']
    with pytest.raises(ValidationError, match="Missing required columns: .*Aemit"):
        data_validator_instance._validate_columns(mock_bigfile_catalog)

def test_validate_attributes_missing(data_validator_instance, mock_bigfile_catalog):
    del mock_bigfile_catalog.attrs['healpix.npix']
    with pytest.raises(ValidationError, match="Missing required attributes: .*healpix.npix"):
        data_validator_instance._validate_attributes(mock_bigfile_catalog)

def test_validate_attributes_inconsistent_edges_offset(data_validator_instance, mock_bigfile_catalog):
    mock_bigfile_catalog.attrs['aemitIndex.edges'] = np.array([0.1, 0.2])
    mock_bigfile_catalog.attrs['aemitIndex.offset'] = np.array([0, 100, 200, 300])
    with pytest.raises(ValidationError, match="Inconsistent edges .* and offset .* lengths"):
        data_validator_instance._validate_attributes(mock_bigfile_catalog)

def test_validate_attributes_non_monotonic_edges(data_validator_instance, mock_bigfile_catalog):
    mock_bigfile_catalog.attrs['aemitIndex.edges'] = np.array([0.1, 0.3, 0.2])
    with pytest.raises(ValidationError, match="aemitIndex.edges must be monotonically increasing"):
        data_validator_instance._validate_attributes(mock_bigfile_catalog)

def test_validate_attributes_non_decreasing_offset(data_validator_instance, mock_bigfile_catalog):
    mock_bigfile_catalog.attrs['aemitIndex.offset'] = np.array([0, 100, 50, 300])
    with pytest.raises(ValidationError, match="aemitIndex.offset must be non-decreasing"):
        data_validator_instance._validate_attributes(mock_bigfile_catalog)

def test_validate_data_consistency_size_mismatch(data_validator_instance, mock_bigfile_catalog):
    mock_bigfile_catalog.size = 200 
    with pytest.raises(ValidationError, match="Data size 200 less than expected 300"):
        data_validator_instance._validate_data_consistency(mock_bigfile_catalog)

@patch("logging.warning")
def test_validate_data_consistency_aemit_warning(mock_log_warning, data_validator_instance, mock_bigfile_catalog):
    invalid_aemit_data = np.array([0.5, 0.0, 0.99])
    mock_bigfile_catalog.__getitem__.return_value.compute.return_value = invalid_aemit_data
    data_validator_instance._validate_data_consistency(mock_bigfile_catalog)
    mock_log_warning.assert_called_with("Found Aemit values outside expected range (0, 1)")

# --- Tests for validate_indices_file ---
@pytest.fixture
def valid_indices_csv(tmp_path):
    file_path = tmp_path / "valid_indices.csv"
    df = pd.DataFrame({'sheet': [0, 1], 'start': [0, 100], 'end': [100, 200]})
    df.to_csv(file_path, index=False)
    return file_path

def test_validate_indices_file_success(data_validator_instance, valid_indices_csv):
    df = data_validator_instance.validate_indices_file(valid_indices_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

def test_validate_indices_file_not_found(data_validator_instance, tmp_path):
    non_existent_file = tmp_path / "no_such_file.csv"
    with pytest.raises(ValidationError, match="Indices file not found"):
        data_validator_instance.validate_indices_file(non_existent_file)

def test_validate_indices_file_read_error(data_validator_instance, tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("sheet,start,end\ninvalid_text_not_csv_format")
    # Mock pandas read_csv to raise an error for this specific file path or content
    with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Error tokenizing data")):
        with pytest.raises(ValidationError, match="Failed to read indices CSV"):
            data_validator_instance.validate_indices_file(bad_csv)

def test_validate_indices_file_missing_columns(data_validator_instance, tmp_path):
    file_path = tmp_path / "missing_cols.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [0]}) # Missing 'end'
    df.to_csv(file_path, index=False)
    with pytest.raises(ValidationError, match="Indices CSV missing columns: .*end"):
        data_validator_instance.validate_indices_file(file_path)

def test_validate_indices_file_wrong_dtype(data_validator_instance, tmp_path):
    file_path = tmp_path / "wrong_dtype.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [0.5], 'end': [100]}) # 'start' is float
    df.to_csv(file_path, index=False)
    with pytest.raises(ValidationError, match="Column 'start' must be integer type"):
        data_validator_instance.validate_indices_file(file_path)

def test_validate_indices_file_invalid_range(data_validator_instance, tmp_path):
    file_path = tmp_path / "invalid_range.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [100], 'end': [0]}) # start >= end
    df.to_csv(file_path, index=False)
    with pytest.raises(ValidationError, match="Invalid ranges in sheets"):
        data_validator_instance.validate_indices_file(file_path)

def test_validate_indices_file_negative_indices(data_validator_instance, tmp_path):
    file_path = tmp_path / "negative_idx.csv"
    df = pd.DataFrame({'sheet': [0], 'start': [-10], 'end': [100]})
    df.to_csv(file_path, index=False)
    with pytest.raises(ValidationError, match="Negative indices in sheets"):
        data_validator_instance.validate_indices_file(file_path)

# --- Tests for validate_output_directory ---
def test_validate_output_directory_exists_writable(data_validator_instance, tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    data_validator_instance.validate_output_directory(out_dir, create=False)

def test_validate_output_directory_not_exists_no_create(data_validator_instance, tmp_path):
    out_dir = tmp_path / "output"
    with pytest.raises(ValidationError, match="Output directory does not exist"):
        data_validator_instance.validate_output_directory(out_dir, create=False)

def test_validate_output_directory_create_success(data_validator_instance, tmp_path):
    out_dir = tmp_path / "output_to_create"
    assert not out_dir.exists()
    data_validator_instance.validate_output_directory(out_dir, create=True)
    assert out_dir.exists()
    assert out_dir.is_dir()

@patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))
def test_validate_output_directory_create_fail(mock_mkdir, data_validator_instance, tmp_path):
    out_dir = tmp_path / "output_fail_create"
    with pytest.raises(ValidationError, match="Cannot create output directory.*Permission denied"):
        data_validator_instance.validate_output_directory(out_dir, create=True)

@patch("pathlib.Path.touch", side_effect=OSError("Permission denied"))
def test_validate_output_directory_not_writable(mock_touch, data_validator_instance, tmp_path):
    out_dir = tmp_path / "not_writable_dir"
    out_dir.mkdir()
    with pytest.raises(ValidationError, match="Output directory not writable.*Permission denied"):
        data_validator_instance.validate_output_directory(out_dir)

# --- Tests for validate_sheet_processing_feasibility ---
@pytest.fixture
def mock_indices_df_for_feasibility():
    return pd.DataFrame({
        'sheet': [0, 1, 2],
        'start': [0, 100, 200],
        'end': [100, 250, 290] 
    })

def test_validate_sheet_processing_feasibility_success(data_validator_instance, mock_bigfile_catalog, mock_indices_df_for_feasibility):
    # mock_bigfile_catalog.size is 300. Sheet 2 ends at 290, which is fine.
    results = data_validator_instance.validate_sheet_processing_feasibility(
        mock_bigfile_catalog, mock_indices_df_for_feasibility
    )
    assert results["feasible"] is True
    assert not results["errors"]
    assert results["stats"]["total_sheets"] == 3

def test_validate_sheet_processing_feasibility_end_exceeds_size(data_validator_instance, mock_bigfile_catalog, mock_indices_df_for_feasibility):
    mock_indices_df_for_feasibility.loc[1, 'end'] = 350 # Sheet 1 end exceeds catalog size 300
    results = data_validator_instance.validate_sheet_processing_feasibility(
        mock_bigfile_catalog, mock_indices_df_for_feasibility
    )
    assert results["feasible"] is False
    assert any("Sheet 1: end index 350 exceeds data size 300" in e for e in results["errors"])

@patch("logging.warning") 
def test_validate_sheet_processing_feasibility_offset_warning(mock_log_warn, data_validator_instance, mock_bigfile_catalog):
    # mock_bigfile_catalog.attrs['aemitIndex.offset'] = [0, 100, 200, 300]
    # For sheet 0, expected_end is offset[0+2] = offset[2] = 200
    # For sheet 1, expected_end is offset[1+2] = offset[3] = 300
    indices_df = pd.DataFrame({'sheet': [0], 'start': [0], 'end': [50]}) # end 50, expected 200. Difference is 150.
                                                                      # The warning threshold is abs(end - expected_end) > 1000
                                                                      # So this should NOT warn.
    results = data_validator_instance.validate_sheet_processing_feasibility(mock_bigfile_catalog, indices_df)
    assert not any("differs significantly" in w for w in results["warnings"])

    # Now make it warn
    mock_bigfile_catalog.attrs['aemitIndex.offset'] = np.array([0, 100, 20000, 30000]) # sheet 0 expected end 20000
    indices_df_warn = pd.DataFrame({'sheet': [0], 'start': [0], 'end': [150]})
    results_warn = data_validator_instance.validate_sheet_processing_feasibility(mock_bigfile_catalog, indices_df_warn)
    assert any("Sheet 0: end index 150 differs significantly from expected 20000" in w for w in results_warn["warnings"])


# --- Tests for validate_healpix_parameters ---
@patch("healpy.npix2nside")
@patch("healpy.nside2resol")
def test_validate_healpix_parameters_success(mock_nside2resol, mock_npix2nside, data_validator_instance):
    npix = 12 * 64**2
    nside = 64
    resolution = 5.0 # Dummy resolution
    mock_npix2nside.return_value = nside
    mock_nside2resol.return_value = resolution

    results = data_validator_instance.validate_healpix_parameters(npix)
    assert results["valid"] is True
    assert results["npix"] == npix
    assert results["nside"] == nside
    assert results["resolution_arcmin"] == resolution
    mock_npix2nside.assert_called_once_with(npix)
    mock_nside2resol.assert_called_once_with(nside, arcmin=True)

@patch("healpy.npix2nside", side_effect=ValueError("Invalid npix for healpy"))
def test_validate_healpix_parameters_invalid_npix_healpy_error(mock_npix2nside, data_validator_instance):
    with pytest.raises(ValidationError, match="Invalid npix value: -100"):
        data_validator_instance.validate_healpix_parameters(-100)

@patch("healpy.npix2nside")
def test_validate_healpix_parameters_nside_not_power_of_2(mock_npix2nside, data_validator_instance):
    npix_for_nside60 = 12 * 60**2 # nside would be 60, not power of 2
    mock_npix2nside.return_value = 60 # Mock nside directly
    with pytest.raises(ValidationError, match="nside 60 is not a power of 2"):
        data_validator_instance.validate_healpix_parameters(npix_for_nside60) 