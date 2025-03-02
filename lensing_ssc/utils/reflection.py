import inspect
from typing import Dict, Any

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