import os
import sys
import numpy as np
import dask.array as da
import healpy as healpix
import gc
import json

import bigfile
import resource
import nbodykit
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
from nbodykit.utils import GatherArray

from mpi4py import MPI
nbodykit.setup_logging()
nbodykit.set_options(dask_chunk_size=1024 * 1024)
nbodykit.set_options(global_cache_size=0)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse

from src.utils.pipeline import CosmicShearPipeline
from src.utils.read_range import read_range

def generate_job_script(z1, z2, config_file, job_template, log_dir, output_dir):
    job_name = "wlen_tiled_config_%02.2f_%02.2f" % (z1, z2)
    job_script = job_template.format(
        job_name=job_name,
        output_file=os.path.join(log_dir, f"{job_name}.out"),
        error_file=os.path.join(log_dir, f"{job_name}.err"),
        config_file=config_file,
        z1=z1,
        z2=z2
    )

    script_filename = os.path.join(output_dir, f"{job_name}.sh")
    with open(script_filename, "w") as script_file:
        script_file.write(job_script)

    print(f"Generated job script: {script_filename}")

    # submit the job
    os.system(f"qsub {script_filename}")

def main(config_file, log_dir, output_dir, job_template):
    # Read the configuration file
    with open(config_file, 'r') as file:
        config = json.load(file)
    # Access the configuration values
    path = config['source']
    dst = config['destination']
    if not os.path.exists(dst):
        os.makedirs(dst)
    cat = BigFileCatalog(path, dataset=config['dataset'])

    zs_list = config['zs']
    zlmin = config['zlmin']
    zlmax = config['zlmax'] if 'zlmax' in config else max(zs_list)
    Nsteps = int(np.round((zlmax - zlmin) / config['zstep']))
    if Nsteps < 2:
        Nsteps = 2
    z = np.linspace(zlmax, zlmin, Nsteps, endpoint=True)

    if cat.comm.rank == 0:
        cat.logger.info("Starting Cosmic Shear Calculation")
        cat.logger.info(f"redshift bins: {z}")

    for z2, z1 in zip(z[1:][::-1], z[:-1][::-1]):
        cat.logger.info("Processing redshift bin %f < z < %f" % (z1, z2))
        sliced = read_range(cat, 1/(1 + z1), 1 / (1 + z2))
        if sliced.csize == 0: continue
        
        # Check if data is already computed
        fname = os.path.join(dst,"kappa_%02.2f_%02.2f.npz" % (z1, z2))
        flag = os.path.exists(fname)
        if flag:
            if cat.comm.rank == 0:
                cat.logger.info("Data already exists for redshift bin %f < z < %f" % (z1, z2))
            continue

        generate_job_script(z1, z2, config_file, job_template, log_dir, output_dir)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--data', type=str, choices=['tiled', 'bigbox'], default='data')
    arg = arg.parse_args()

    config_file = f"/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_{arg.data}.json"
    log_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/job/log"
    output_dir = f"/lustre/work/akira.tokiwa/Projects/LensingSSC/job/scripts/config_{arg.data}"
    job_template = """#!/bin/bash
#PBS -N {job_name}
#PBS -o {output_file}
#PBS -e {error_file}
#PBS -l nodes=4:ppn=8,walltime=99999:99:99
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate cfastpm
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/
MPI=/opt/intel/psxe2018/compilers_and_libraries_2018.5.274/linux/mpi/intel64/bin/mpiexec.hydra
CFG={config_file}
$MPI -n 32 python -m src.rsbin.wlen_zbin $CFG {z1} {z2}
    """
    main(config_file, log_dir, output_dir, job_template)