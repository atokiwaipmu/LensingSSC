import argparse

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