import argparse
import inspect
import yaml
import logging
from typing import Any, Dict

def filter_config(config: Dict[str, Any], cls: object) -> Dict[str, Any]:
    """
    Filter out configurations that do not match the initialization parameters
    of a given class.

    Parameters
    ----------
    config : dict
        A dictionary of configurations.
    cls : object
        The class whose constructor parameters will be used to filter 'config'.

    Returns
    -------
    dict
        A filtered dictionary containing only the parameters accepted by cls.__init__.
    """
    parameters = inspect.signature(cls.__init__).parameters
    filtered_config = {key: val for key, val in config.items() if key in parameters}
    return filtered_config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datadir",
        type=str,
        help="Directory containing data"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing"
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    SystemExit
        If the YAML file cannot be parsed or read.
    """
    try:
        with open(config_file, "r") as file_stream:
            config = yaml.safe_load(file_stream)
        return config
    except (yaml.YAMLError, OSError) as exc:
        logging.error(f"Failed to load config file '{config_file}': {exc}")
        raise SystemExit(1) from exc


def setup_logging() -> None:
    """
    Configure the logging settings for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

