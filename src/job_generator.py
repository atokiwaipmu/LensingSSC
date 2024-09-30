
import subprocess
from src.info_extractor import InfoExtractor

class JobGenerator:
    def __init__(self, datadir, 
                 config_file="/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_default.yaml", 
                 workdir="/lustre/work/akira.tokiwa/Projects/LensingSSC/", 
                 user="akira.tokiwa",
                 env_name="lssc",
                 queue="mini",
                 ppn=48):
        info = InfoExtractor.extract_info_from_path(datadir)
        self.datadir = datadir
        self.config_file = config_file
        self.seed = info["seed"]
        self.box_type = info["box_type"]

        self.workdir = workdir
        self.user = user
        self.mail = f"{user}.ipmu.jp"
        self.env_name = env_name
        self.queue = queue
        self.ppn = ppn

    def gen_script(self, task, job_name, option=None, if_omp=False, if_submit=False):
        if if_omp:
            omp_export = "export OMP_NUM_THREADS=48"
        else:
            omp_export = ""

        script_template = f"""#!/bin/bash
#PBS -N {job_name}_{self.box_type}_s{self.seed}
#PBS -o {self.datadir}/log/{job_name}_{self.box_type}_s{self.seed}.out
#PBS -e {self.datadir}/log/{job_name}_{self.box_type}_s{self.seed}.err
#PBS -l nodes=1:ppn={self.ppn},walltime=10:00:00
#PBS -u {self.user}
#PBS -M {self.mail}
#PBS -m ae
#PBS -q {self.queue}

source ~/.bashrc
conda activate {self.env_name}
cd {self.workdir}

{omp_export}

DATA_DIR={self.datadir}
CONFIG_FILE={self.config_file}
python -m {task} $DATA_DIR $CONFIG_FILE {option if option else ""}
"""

        with open(f"{self.datadir}/{job_name}.sh", "w") as file:
            file.write(script_template)

        if if_submit:
            subprocess.run(["qsub", f"{self.datadir}/{job_name}.sh"])