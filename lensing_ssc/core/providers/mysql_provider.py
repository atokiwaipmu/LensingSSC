import mysql.connector
from mysql.connector import errorcode
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple

# Assuming StorageProvider is in this path based on previous context
from lensing_ssc.core.interfaces.storage_interface import StorageProvider


class MySQLProvider(StorageProvider):
    def __init__(self, config=None):
        # super().__init__() # StorageProvider might not have an __init__ if it's purely an interface
        self._logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.connection = None
        self._initialized = False
        self._connector_version = "unknown"

        if self.config:
            self.initialize(**self.config) # Auto-initialize if config is provided

    def _check_dependencies(self) -> None:
        """Check if mysql.connector is installed and get version."""
        try:
            import mysql.connector
            self._connector_version = mysql.connector.__version__
        except ImportError:
            self._connector_version = "unavailable"
            raise ImportError(
                "MySQLProvider requires the 'mysql-connector-python' package. "
                "Please install it using 'pip install mysql-connector-python'."
            )

    def initialize(self, **kwargs):
        """Initialize connection to MySQL."""
        if self._initialized:
            self._logger.debug("MySQLProvider already initialized.")
            return

        # If config was passed to __init__, use it. Otherwise, expect it in kwargs.
        config_to_use = self.config if self.config else kwargs
        if not config_to_use:
            self._logger.error("MySQL configuration not provided for initialization.")
            raise ValueError("MySQL configuration not provided")

        self._check_dependencies() # Ensure dependencies are met before connecting

        try:
            self.connection = mysql.connector.connect(**config_to_use)
            self._logger.info("MySQL connection established.")
            self._initialized = True
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                self._logger.error("MySQL connection error: Access denied (user/password).")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                # This can happen if the database in connection string doesn't exist
                # For some operations (like creating a database), we might not need a db in connection string.
                self._logger.warning(f"MySQL connection warning: Database '{config_to_use.get('database')}' may not exist.")
                # For basic operations like SHOW DATABASES, a connection without specifying a DB is fine.
                # If a DB was specified and it's bad, we might still want to connect to server.
                temp_config = config_to_use.copy()
                db_name = temp_config.pop('database', None)
                try:
                    self.connection = mysql.connector.connect(**temp_config)
                    self._logger.info(f"MySQL connection established to server (database '{db_name}' not selected or not found).")
                    self._initialized = True
                except mysql.connector.Error as server_err:
                    self._logger.error(f"MySQL server connection error: {server_err}")
                    raise server_err # Re-raise if server connection fails
            else:
                self._logger.error(f"MySQL connection error: {err}")
            if not self._initialized: # If connection failed or partially failed
                 raise err # Re-raise the original error or a new one

    def shutdown(self):
        """Close the MySQL connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self._logger.info("MySQL connection is closed")
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            if self.config:
                self.initialize(**self.config)
            else:
                # This state implies config was not provided at init or via initialize method
                self._logger.error("MySQLProvider not initialized and no configuration available.")
                raise ConnectionError("MySQLProvider not initialized. Call initialize() with config first.")

    @property
    def name(self) -> str:
        return "MySQLProvider"

    @property
    def version(self) -> str:
        if self._connector_version == "unknown" or self._connector_version == "unavailable":
            try:
                self._check_dependencies()
            except ImportError:
                pass # _connector_version remains "unavailable"
        return self._connector_version

    def is_available(self) -> bool:
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False

    def _parse_path(self, path: Union[str, Path]) -> Tuple[str, Optional[str]]:
        """Parses path into (database_name, table_name_or_none)."""
        path_str = str(path)
        if '/' in path_str:
            db_name, table_name = path_str.split('/', 1)
            return db_name.strip(), table_name.strip()
        return path_str.strip(), None

    def execute_query(self, query: str, params: Optional[tuple] = None, database: Optional[str] = None):
        self._ensure_initialized()
        # Temporarily switch database if specified and different from current
        current_db = self.connection.database
        if database and database != current_db:
            try:
                self.connection.database = database
            except mysql.connector.Error as err:
                self._logger.error(f"Failed to switch to database '{database}': {err}")
                raise

        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")):
                self.connection.commit()
            return cursor
        except mysql.connector.Error as err:
            self._logger.error(f"Error executing query: {err}\nQuery: {query}")
            self.connection.rollback()
            raise # Re-raise after logging
        finally:
            cursor.close()
            # Restore previous database if it was changed
            if database and database != current_db and current_db is not None : # and self.connection.is_connected()
                try: # Check if connection is still valid before trying to set db
                    if self.connection.is_connected():
                         self.connection.database = current_db
                except mysql.connector.Error: # pragma: no cover
                    self._logger.warning(f"Could not restore database to '{current_db}'. Connection might be lost.")


    def fetch_data(self, query: str, params: Optional[tuple] = None, database: Optional[str] = None) -> List[tuple]:
        cursor = self.execute_query(query, params, database=database)
        return cursor.fetchall()

    # --- StorageProvider Interface Implementation ---

    def exists(self, path: Union[str, Path]) -> bool:
        self._ensure_initialized()
        db_name, table_name = self._parse_path(path)

        if table_name:  # Check for table
            try:
                # Ensure connection is to the correct database
                if self.connection.database != db_name:
                    # Check if db_name exists before trying to switch
                    all_dbs = [row[0] for row in self.fetch_data("SHOW DATABASES;")]
                    if db_name not in all_dbs:
                        return False # Database itself doesn't exist
                    self.connection.database = db_name

                tables = self.fetch_data(f"SHOW TABLES LIKE '{table_name}';")
                return len(tables) > 0
            except mysql.connector.Error as err:
                # This can happen if db_name does not exist or access denied
                self._logger.warning(f"Could not check existence of table '{table_name}' in db '{db_name}': {err}")
                return False
        else:  # Check for database
            try:
                databases = self.fetch_data("SHOW DATABASES;")
                return any(db_name == db[0] for db in databases)
            except mysql.connector.Error as err:
                self._logger.error(f"Error checking database existence for '{db_name}': {err}")
                return False


    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        self._ensure_initialized()
        db_name, table_name = self._parse_path(path)

        if table_name:
            self._logger.warning(f"mkdir for table '{db_name}/{table_name}' is not implemented. Use specific table creation methods.")
            raise NotImplementedError("Direct table creation via mkdir requires schema definition and is not supported. Use execute_query with CREATE TABLE statement.")

        # Create database
        try:
            if self.exists(db_name):
                if exist_ok:
                    self._logger.info(f"Database '{db_name}' already exists.")
                    return
                else:
                    raise FileExistsError(f"Database '{db_name}' already exists.")

            self.execute_query(f"CREATE DATABASE `{db_name}`;")
            self._logger.info(f"Database '{db_name}' created successfully.")
        except mysql.connector.Error as err:
            self._logger.error(f"Failed to create database '{db_name}': {err}")
            raise # Re-raise to indicate failure

    def remove(self, path: Union[str, Path]) -> None:
        self._ensure_initialized()
        db_name, table_name = self._parse_path(path)

        if not self.exists(path):
            self._logger.warning(f"Cannot remove '{path}': Does not exist.")
            # According to os.remove, FileNotFoundError should be raised.
            raise FileNotFoundError(f"No such database or table: '{path}'")

        try:
            if table_name: # Drop table
                self.execute_query(f"DROP TABLE `{db_name}`.`{table_name}`;", database=db_name)
                self._logger.info(f"Table '{db_name}/{table_name}' removed successfully.")
            else: # Drop database
                self.execute_query(f"DROP DATABASE `{db_name}`;")
                self._logger.info(f"Database '{db_name}' removed successfully.")
        except mysql.connector.Error as err:
            self._logger.error(f"Failed to remove '{path}': {err}")
            raise # Re-raise to indicate failure

    def list_files(self, path: Union[str, Path], pattern: Optional[str] = None) -> List[Path]:
        self._ensure_initialized()
        db_name, table_name = self._parse_path(path)

        if table_name:
            self._logger.warning(f"list_files for table '{db_name}/{table_name}' is not applicable. It lists tables within a database.")
            return [] # Or raise error, depending on desired strictness

        if not self.exists(db_name):
            raise FileNotFoundError(f"Database '{db_name}' not found.")

        query = "SHOW TABLES;"
        if pattern:
            query = f"SHOW TABLES LIKE '{pattern}';"

        try:
            tables_tuples = self.fetch_data(query, database=db_name)
            # Path objects for tables should probably be db_name/table_name
            return [Path(f"{db_name}/{row[0]}") for row in tables_tuples]
        except mysql.connector.Error as err:
            self._logger.error(f"Failed to list tables in database '{db_name}': {err}")
            return []

    def __enter__(self):
        # Ensure initialized when entering context
        if not self._initialized and self.config:
            self.initialize(**self.config)
        elif not self._initialized:
             # This case means config was not provided at construction
             # Let _ensure_initialized handle it if methods are called
             pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

```
