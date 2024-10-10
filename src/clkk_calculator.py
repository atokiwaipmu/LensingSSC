# src/clkk_calculator.py

import argparse
import healpy as hp
import logging
from pathlib import Path

def process_map(kappa_path: Path, clkk_dir: Path, lmax: int, overwrite: bool) -> None:
    """
    Process a single kappa map to compute and save its Cl_kk power spectrum.

    Args:
        kappa_path (Path): Path to the input kappa map.
        clkk_dir (Path): Directory to save the Cl_kk output.
        lmax (int): The maximum multipole moment.
        overwrite (bool): Whether to overwrite existing files.
    """
    output_filename = kappa_path.name.replace("kappa", "cl")
    if "ngal" not in output_filename:
        output_filename = output_filename.replace(".fits", "_noiseless.fits")
    output_path = clkk_dir / output_filename

    if output_path.exists() and not overwrite:
        logging.info(f"Skipping {output_path.name} (already exists).")
        return

    try:
        logging.info(f"Processing {kappa_path.name}")
        kappa_map = hp.read_map(str(kappa_path))
        kappa_map = hp.reorder(kappa_map, n2r=True)
        logging.info(f"Loaded kappa map with {len(kappa_map)} pixels.")
        clkk = hp.anafast(kappa_map, lmax=lmax)
        logging.info(f"Computed Cl_kk with {len(clkk)} multipoles.")
        hp.write_cl(str(output_path), clkk, overwrite=overwrite)
        logging.info(f"Saved {output_path.name}")
    except Exception as e:
        logging.error(f"Failed to process {kappa_path.name}: {e}")

def process_datadir(datadir: Path, lmax: int = 3000, overwrite: bool = False) -> None:
    """
    Process all kappa and noisy maps within a single data directory.

    Args:
        datadir (Path): The data directory containing kappa and noisy_maps folders.
        lmax (int): The maximum multipole moment.
        overwrite (bool): Whether to overwrite existing files.
    """
    kappa_dir = datadir / "kappa"
    noisy_dir = datadir / "noisy_maps"

    # Collect all relevant map paths
    kappa_paths = sorted(kappa_dir.glob("*.fits")) if kappa_dir.exists() else []
    noisy_paths = sorted(noisy_dir.glob("*.fits")) if noisy_dir.exists() else []

    if not kappa_paths and not noisy_paths:
        logging.warning(f"No kappa or noisy maps found in {datadir}. Skipping.")
        return

    clkk_dir = datadir / "cls"
    clkk_dir.mkdir(exist_ok=True, parents=True)

    for kappa_path in kappa_paths:
        process_map(kappa_path, clkk_dir, lmax, overwrite)

    #for noisy_path in noisy_paths:
    #    process_map(noisy_path, clkk_dir, lmax, overwrite)

if __name__ == "__main__":
    from src.utils import parse_arguments, load_config, setup_logging
    setup_logging()
    args = parse_arguments()
    config = load_config(args.config_file)

    process_datadir(Path(args.datadir), lmax=config.get("lmax", 3000), overwrite=args.overwrite)