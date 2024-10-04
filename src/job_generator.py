# src/job_generator.py

import subprocess
from dataclasses import dataclass, field
from src.info_extractor import InfoExtractor

@dataclass
class JobGenerator:
    datadir: str
    config_file: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_default.yaml"
    workdir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/"
    user: str = "akira.tokiwa"
    env_name: str = "lssc"
    queue: str = "mini"
    ppn: int = 48
    mail: str = field(init=False)
    seed: str = field(init=False)
    box_type: str = field(init=False)

    def __post_init__(self):
        info = InfoExtractor.extract_info_from_path(self.datadir)
        self.seed = info["seed"]
        self.box_type = info["box_type"]
        self.mail = f"{self.user}@ipmu.jp"

    def gen_script(self, task, job_name, option=None, if_omp=False, if_submit=False):
        omp_export = f"export OMP_NUM_THREADS={self.ppn}" if if_omp else ""
        script_content = f"""#!/bin/bash
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
python -m {task} $DATA_DIR $CONFIG_FILE {option or ""}
"""
        script_path = f"{self.datadir}/{job_name}.sh"
        with open(script_path, "w") as file:
            file.write(script_content)

        if if_submit:
            subprocess.run(["qsub", script_path])
