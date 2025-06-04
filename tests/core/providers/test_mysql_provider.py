import pytest
from unittest import mock
from pathlib import Path

from lensing_ssc.core.providers.mysql_provider import MySQLProvider
from lensing_ssc.core.interfaces.storage_interface import StorageProvider
# Import mysql.connector.errors for specific error types if needed for more advanced mocking
from mysql.connector import errors as mysql_errors


# --- Configuration for Mock Testing ---
# Mock database configuration
MOCK_DB_CONFIG = {
    'host': 'localhost',
    'user': 'testuser',
    'password': 'testpassword',
    # 'database': 'test_db' # Usually specified in path for storage operations
}

MOCK_DB_NAME = "test_db"
MOCK_TABLE_NAME = "test_table"

# --- Pytest Fixtures ---

@pytest.fixture
def mock_mysql_connector():
    """Mocks the mysql.connector library."""
    with mock.patch('lensing_ssc.core.providers.mysql_provider.mysql.connector') as mock_connector:
        # Mock connection object
        mock_connection = mock.Mock()
        mock_connection.is_connected.return_value = True
        mock_connection.database = None # Current database on connection

        # Mock cursor object
        mock_cursor = mock.Mock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        mock_cursor.rowcount = 0

        mock_connection.cursor.return_value = mock_cursor
        mock_connector.connect.return_value = mock_connection

        # Mock version attribute
        mock_connector.__version__ = "8.0.28.mock"

        # Store mocks for assertions
        mock_connector._mock_connection = mock_connection
        mock_connector._mock_cursor = mock_cursor
        yield mock_connector

@pytest.fixture
def mysql_provider_mocked(mock_mysql_connector):
    """Provides a MySQLProvider instance with a mocked mysql.connector."""
    # Initialize with a base config, actual db specified in methods
    provider = MySQLProvider(config=MOCK_DB_CONFIG)
    # Ensure that the initialization within provider uses the mock
    # The __init__ of MySQLProvider calls self.initialize if config is present.
    # If not, it might be called later.
    # If initialize was already called by __init__, this re-call might be redundant
    # but ensures it uses the mocked connector if not already.
    # provider.initialize(**MOCK_DB_CONFIG) # Already called in __init__
    return provider

@pytest.fixture
def mysql_provider_uninitialized_mocked(mock_mysql_connector):
    """Provides a MySQLProvider instance without initial config, using mocked connector."""
    provider = MySQLProvider(config=None) # No initial config
    return provider


# --- Test Cases ---

class TestMySQLProviderMocked:
    """Test suite for MySQLProvider using a mocked mysql.connector."""

    def test_provider_identity(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        assert mysql_provider_mocked.name == "MySQLProvider"
        assert mysql_provider_mocked.version == "8.0.28.mock"
        assert mysql_provider_mocked.is_available() is True

    def test_is_available_failure(self, mysql_provider_uninitialized_mocked: MySQLProvider, mock_mysql_connector):
        # Test when mysql.connector cannot be imported
        mock_mysql_connector.reset_mock() # Reset any previous calls
        # Simulate ImportError when _check_dependencies is called
        original_import = lensing_ssc.core.providers.mysql_provider.mysql.connector

        # Temporarily break the import
        with mock.patch('lensing_ssc.core.providers.mysql_provider.mysql.connector', None):
            # Need to create a new provider for its _check_dependencies to run again in is_available
            # or re-evaluate version.
            # Forcing version re-evaluation
            provider = MySQLProvider(config=None) # New instance
            provider._connector_version = "unknown" # Reset cached version
            assert provider.is_available() is False
            assert provider.version == "unavailable"

        # Restore for other tests
        lensing_ssc.core.providers.mysql_provider.mysql.connector = original_import


    def test_connection_management_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        # Connection should have been established by fixture's __init__
        mock_mysql_connector.connect.assert_called_once_with(**MOCK_DB_CONFIG)
        assert mysql_provider_mocked._initialized is True

        mysql_provider_mocked.shutdown()
        mock_mysql_connector._mock_connection.close.assert_called_once()
        assert mysql_provider_mocked._initialized is False

    def test_connection_failure_mocked(self, mock_mysql_connector):
        mock_mysql_connector.connect.side_effect = mysql_errors.InterfaceError("Connection failed")
        with pytest.raises(mysql_errors.InterfaceError):
            # Need a new provider instance to trigger connect attempt in __init__
            MySQLProvider(config=MOCK_DB_CONFIG)

    def test_database_exists_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Test case: database exists
        mock_cursor.fetchall.return_value = [(MOCK_DB_NAME,), ("another_db",)]
        assert mysql_provider_mocked.exists(MOCK_DB_NAME) is True
        mock_cursor.execute.assert_called_with("SHOW DATABASES;")

        # Test case: database does not exist
        mock_cursor.fetchall.return_value = [("another_db",)]
        assert mysql_provider_mocked.exists("non_existent_db") is False

    def test_database_create_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Simulate database does not exist initially
        mock_cursor.fetchall.return_value = []

        mysql_provider_mocked.mkdir(MOCK_DB_NAME)
        # First call to SHOW DATABASES in exists(), second in mkdir for CREATE DB
        # execute("SHOW DATABASES;") in exists(), then execute("CREATE DATABASE...")
        # Check for the CREATE DATABASE call
        assert any(call == mock.call(f"CREATE DATABASE `{MOCK_DB_NAME}`;") for call in mock_cursor.execute.call_args_list)

        # Test exist_ok=True
        mock_cursor.fetchall.return_value = [(MOCK_DB_NAME,)] # Simulate DB now exists
        mysql_provider_mocked.mkdir(MOCK_DB_NAME, exist_ok=True) # Should not raise

        # Test exist_ok=False
        with pytest.raises(FileExistsError):
            mysql_provider_mocked.mkdir(MOCK_DB_NAME, exist_ok=False)

    def test_database_remove_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Simulate database exists
        mock_cursor.fetchall.return_value = [(MOCK_DB_NAME,)]

        mysql_provider_mocked.remove(MOCK_DB_NAME)
        assert any(call == mock.call(f"DROP DATABASE `{MOCK_DB_NAME}`;") for call in mock_cursor.execute.call_args_list)

        # Test remove non-existent
        mock_cursor.fetchall.return_value = [] # Simulate DB does not exist
        with pytest.raises(FileNotFoundError):
            mysql_provider_mocked.remove("non_existent_db")

    def test_table_exists_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor
        mock_conn = mock_mysql_connector._mock_connection

        # Simulate database exists, then table exists
        # 1. exists(db_name) call for SHOW DATABASES
        # 2. exists(db_name/table_name) call for SHOW TABLES
        mock_cursor.fetchall.side_effect = [
            [(MOCK_DB_NAME,)],  # For SHOW DATABASES (db exists)
            [(MOCK_TABLE_NAME,)] # For SHOW TABLES LIKE 'test_table'
        ]
        assert mysql_provider_mocked.exists(f"{MOCK_DB_NAME}/{MOCK_TABLE_NAME}") is True
        mock_conn.database = MOCK_DB_NAME # Connection should switch to MOCK_DB_NAME
        # Correct query for SHOW TABLES LIKE
        mock_cursor.execute.assert_any_call(f"SHOW TABLES LIKE '{MOCK_TABLE_NAME}';")


        # Simulate table does not exist
        mock_cursor.fetchall.side_effect = [
            [(MOCK_DB_NAME,)], # For SHOW DATABASES
            []                 # For SHOW TABLES LIKE 'non_existent_table'
        ]
        assert mysql_provider_mocked.exists(f"{MOCK_DB_NAME}/non_existent_table") is False

    def test_table_remove_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Simulate DB and table exist
        mock_cursor.fetchall.side_effect = [
            [(MOCK_DB_NAME,)],      # exists(db_name) in remove path
            [(MOCK_TABLE_NAME,)]    # exists(table_name) in remove path
        ]

        path_str = f"{MOCK_DB_NAME}/{MOCK_TABLE_NAME}"
        mysql_provider_mocked.remove(path_str)
        # Check for the DROP TABLE call
        # The database context should be set before this query
        expected_query = f"DROP TABLE `{MOCK_DB_NAME}`.`{MOCK_TABLE_NAME}`;"

        # Verify execute was called with the DROP TABLE query
        # We need to check all calls to execute because it's called multiple times
        # by .exists() as well.
        was_called = False
        for call_args in mock_cursor.execute.call_args_list:
            if call_args[0][0] == expected_query:
                was_called = True
                break
        assert was_called, f"Expected query '{expected_query}' not found in execute calls."
        assert mock_mysql_connector._mock_connection.database == MOCK_DB_NAME

    def test_list_tables_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Simulate database exists
        mock_cursor.fetchall.side_effect = [
            [(MOCK_DB_NAME,)], # For exists(MOCK_DB_NAME)
            [("table1",), ("table2",), ("another_table",)] # For SHOW TABLES
        ]

        tables = mysql_provider_mocked.list_files(MOCK_DB_NAME)
        assert tables == [Path(f"{MOCK_DB_NAME}/table1"), Path(f"{MOCK_DB_NAME}/table2"), Path(f"{MOCK_DB_NAME}/another_table")]
        mock_cursor.execute.assert_called_with("SHOW TABLES;") # Last relevant call

        # Test with pattern
        mock_cursor.fetchall.side_effect = [
            [(MOCK_DB_NAME,)], # For exists(MOCK_DB_NAME)
            [("table1",), ("table2",)] # For SHOW TABLES LIKE 'table%'
        ]
        pattern = "table%"
        tables_pattern = mysql_provider_mocked.list_files(MOCK_DB_NAME, pattern=pattern)
        assert tables_pattern == [Path(f"{MOCK_DB_NAME}/table1"), Path(f"{MOCK_DB_NAME}/table2")]
        mock_cursor.execute.assert_called_with(f"SHOW TABLES LIKE '{pattern}';")

    def test_list_files_non_existent_db_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor
        # Simulate database does NOT exist
        mock_cursor.fetchall.return_value = []
        with pytest.raises(FileNotFoundError):
            mysql_provider_mocked.list_files("non_existent_db")

    def test_mkdir_for_table_raises_not_implemented(self, mysql_provider_mocked: MySQLProvider):
        with pytest.raises(NotImplementedError):
            mysql_provider_mocked.mkdir(f"{MOCK_DB_NAME}/{MOCK_TABLE_NAME}")

    def test_execute_and_fetch_data_mocked(self, mysql_provider_mocked: MySQLProvider, mock_mysql_connector):
        mock_cursor = mock_mysql_connector._mock_cursor

        # Test execute_query for a DDL-like command (no results expected from cursor.fetchall)
        ddl_query = "CREATE TABLE mock_table (id INT);"
        mysql_provider_mocked.execute_query(ddl_query, database=MOCK_DB_NAME)
        mock_cursor.execute.assert_called_with(ddl_query, None)
        mock_mysql_connector._mock_connection.commit.assert_called_once()

        mock_mysql_connector._mock_connection.commit.reset_mock() # Reset for next call

        # Test execute_query for DML (e.g. INSERT)
        dml_query = "INSERT INTO mock_table (id) VALUES (%s);"
        params = (1,)
        mysql_provider_mocked.execute_query(dml_query, params, database=MOCK_DB_NAME)
        mock_cursor.execute.assert_called_with(dml_query, params)
        mock_mysql_connector._mock_connection.commit.assert_called_once()

        # Test fetch_data
        select_query = "SELECT * FROM mock_table;"
        expected_data = [(1,), (2,)]
        mock_cursor.fetchall.return_value = expected_data

        data = mysql_provider_mocked.fetch_data(select_query, database=MOCK_DB_NAME)
        mock_cursor.execute.assert_called_with(select_query, None)
        assert data == expected_data

    def test_context_manager_mocked(self, mock_mysql_connector):
        mock_conn = mock_mysql_connector._mock_connection
        with MySQLProvider(config=MOCK_DB_CONFIG) as provider:
            assert provider._initialized is True
            mock_mysql_connector.connect.assert_called_with(**MOCK_DB_CONFIG)

        mock_conn.close.assert_called_once()
        assert provider._initialized is False # Check status after exit

    def test_initialize_without_config_in_init(self, mysql_provider_uninitialized_mocked: MySQLProvider, mock_mysql_connector):
        provider = mysql_provider_uninitialized_mocked
        assert provider._initialized is False

        # Call methods that trigger _ensure_initialized
        with pytest.raises(ConnectionError, match="MySQLProvider not initialized. Call initialize() with config first."):
            provider.exists("some_db")

        # Now initialize
        provider.initialize(**MOCK_DB_CONFIG)
        mock_mysql_connector.connect.assert_called_once_with(**MOCK_DB_CONFIG)
        assert provider._initialized is True

        # Subsequent calls should work
        mock_mysql_connector._mock_cursor.fetchall.return_value = []
        provider.exists("some_db")
        # ensure execute was called (it is by exists)
        assert mock_mysql_connector._mock_cursor.execute.call_count > 0

# --- Placeholder for Live DB Tests ---
# @pytest.mark.skip(reason="Requires a live MySQL database instance and configuration.")
# class TestMySQLProviderLive:
#     # Actual test methods requiring a live DB would go here.
#     # These would need a fixture that sets up a real test DB.
#     pass

# --- Final Note ---
# print("\nNOTE: Most MySQLProvider tests are run using a mocked mysql.connector.")
# print("Full integration tests (e.g., for complex queries or transaction behavior) would require a live MySQL database.")
# print("A placeholder class TestMySQLProviderLive is included but skipped.")
# print("To run live tests, a proper fixture managing a test database is needed,")
# print("and test database credentials should be securely managed (e.g., via environment variables).")

# It's good practice to make sure the module can be imported and basic fixtures work
def test_module_import():
    assert True
