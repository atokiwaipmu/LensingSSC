import yaml
import logging
from typing import Dict, Any

def load_config(config_file: str) -> Dict[str, Any]:
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