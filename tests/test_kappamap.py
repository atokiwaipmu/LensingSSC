import argparse
import logging
import os
import warnings

import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI

from src.masssheet.ConfigData import ConfigData
from src.masssheet.kappamap import SheetMapper, wlen_chi_kappa, load_delta_sheet
from src.masssheet.kappamap import main as kappamap_main

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    parser.add_argument('--i_start', type=int, default=40, help='Start index of mass sheet')
    parser.add_argument('--i_end', type=int, default=41, help='End index of mass sheet')
    parser.add_argument('--r4096', type=bool, default=False, help='Use 4096 resolution')

    args = parser.parse_args()

    config_file = os.path.join(
        "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs",
        f'config_{args.config}_hp.json'
    )
    config = ConfigData.from_json(config_file)

    data_path = os.path.join(config.datadir, "mass_sheets")
    save_path = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/results/test", args.config)
    if args.r4096:
        save_path += "-4096"
    os.makedirs(save_path, exist_ok=True)

    zs = config.zs_list[0]
    kappamap_main(data_path, save_path, zs, i_start=args.i_start, i_end=args.i_end, r4096=args.r4096)
