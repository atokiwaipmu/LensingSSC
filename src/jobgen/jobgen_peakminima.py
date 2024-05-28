import os

from src.masssheet.ConfigData import ConfigData

def generate_job_script(redshift: float, job_dir: str, log_dir: str, script_dir: str, user: str, email: str, config: str):
    job_script = f"""#!/bin/bash
#PBS -N PeaksMinima_{config}_{redshift:.1f}
#PBS -o {log_dir}/PeaksMinima_{config}_{redshift:.1f}.out
#PBS -e {log_dir}/PeaksMinima_{config}_{redshift:.1f}.err
#PBS -l nodes=1:ppn=52,walltime=24:00:00
#PBS -u {user}
#PBS -M {email}
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/
python -m src.analysis.PeaksMinima {config} {redshift}
"""

    script_filename = os.path.join(script_dir, f'job_script_{redshift:.1f}.sh')
    with open(script_filename, 'w') as file:
        file.write(job_script)
    
    print(f"Generated job script for redshift {redshift:.1f}: {script_filename}")

def main():
    job_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/job"
    user = "akira.tokiwa"
    email = "akira.tokiwa@ipmu.jp"

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)
    
    redshifts = config_data.zs_list
    for config in ['tiled', 'bigbox']:
        log_dir = os.path.join(job_dir, "log", f"peakminima_{config}")
        script_dir = os.path.join(job_dir, "scripts", f"peakminima_{config}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)
        for z in redshifts:
            generate_job_script(z, job_dir, log_dir, script_dir, user, email, config)

if __name__ == "__main__":
    main()
