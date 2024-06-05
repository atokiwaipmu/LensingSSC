
import os
from typing import List

def generate_job_script(script_filename: str,
                        jobname: str, project_dir: str, ppn: int, queue: str, relative_path: str, args: str):
    job_script = f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o {project_dir}/log/{jobname}.out
#PBS -e {project_dir}/log/{jobname}.err
#PBS -l nodes=1:ppn={ppn},walltime=24:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa.ipmu.jp
#PBS -m ae
#PBS -q {queue}

source ~/.bashrc
conda activate lssc
cd {project_dir}
python -m {relative_path} {args}
""".format(jobname=jobname, project_dir=project_dir, ppn=ppn, queue=queue, relative_path=relative_path, args=args)

    with open(script_filename, 'w') as file:
        file.write(job_script)

    return 

def generate_submission_script(project_dir: str, jobname: str, scripts: List[str]):
    # generate a file to submit all the scripts
    submit_filename = os.path.join(project_dir, "job", "submission", f"submit_{jobname}.sh")
    with open(submit_filename, 'w') as file:
        for script in scripts:
            file.write(f"qsub {script}\n")

    return submit_filename

