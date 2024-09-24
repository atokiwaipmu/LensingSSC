
import os
import subprocess

from src.analysis import SuffixGenerator

def gen_script(datadir, seed_number):
    script_template = f"""#!/bin/bash
#PBS -N preanalysis_s{seed_number}
#PBS -o {datadir}/log/preanalysis_s{seed_number}.out
#PBS -e {datadir}/log/preanalysis_s{seed_number}.err
#PBS -l nodes=1:ppn=48,walltime=10:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q mini

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

export OMP_NUM_THREADS=48

DATA_DIR={datadir}
CONFIG_FILE=/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_default.yaml
python -m src.preanalysis $DATA_DIR $CONFIG_FILE 
"""

    with open(f"{datadir}/preanalysis.sh", "w") as file:
        file.write(script_template)

def process_subdir(subdir_path):
    seed_number = SuffixGenerator.extract_seed_from_path(subdir_path)
    if seed_number is not None:
        gen_script(subdir_path, seed_number)

def main(main_directory):
    # Traverse subdirectories
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            print(os.path.basename(subdir_path))
            process_subdir(subdir_path)
            subprocess.run(["qsub", f"{subdir_path}/preanalysis.sh"])

if __name__ == "__main__":
    main("/lustre/work/akira.tokiwa/Projects/LensingSSC/data/tiled")
    #main("/lustre/work/akira.tokiwa/Projects/LensingSSC/data/bigbox")
    