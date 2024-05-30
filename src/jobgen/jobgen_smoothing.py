import os

from src.masssheet.ConfigData import ConfigData, ConfigAnalysis

def generate_job_script(redshift: float, smoothing_length: float, job_dir: str, log_dir: str, script_dir: str, user: str, email: str, config: str):
    job_script = f"""#!/bin/bash
#PBS -N smoothing_{config}_{redshift:.1f}_{smoothing_length}
#PBS -o {log_dir}/smoothing_{config}_{redshift:.1f}_{smoothing_length}.out
#PBS -e {log_dir}/smoothing_{config}_{redshift:.1f}_{smoothing_length}.err
#PBS -l nodes=1:ppn=4,walltime=24:00:00
#PBS -u {user}
#PBS -M {email}
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/
python -m src.analysis.map_smoothing {config} {redshift} {smoothing_length}
"""

    script_filename = os.path.join(script_dir, f'job_smoothing_{redshift:.1f}_{smoothing_length}.sh')
    with open(script_filename, 'w') as file:
        file.write(job_script)
    
    print(f"Generated job script for redshift {redshift:.1f} and smoothing length {smoothing_length}: {script_filename}")
    return script_filename

def main():
    job_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/job"
    user = "akira.tokiwa"
    email = "akira.tokiwa@ipmu.jp"

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)
    
    scripts = []
    for config in ['tiled', 'bigbox']:
        log_dir = os.path.join(job_dir, "log", f"smoothing_{config}")
        script_dir = os.path.join(job_dir, "scripts", f"smoothing_{config}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)
        for z in config_data.zs_list:
            for sl in config_analysis.sl_arcmin:
                script_filename = generate_job_script(z, sl, job_dir, log_dir, script_dir, user, email, config)
                scripts.append(script_filename)

    # generate a file to submit all the scripts
    submit_filename = os.path.join(job_dir, "submission", "submit_smoothing.sh")
    with open(submit_filename, 'w') as file:
        for script in scripts:
            file.write(f"qsub {script}\n")
    print(f"Generated submit script: {submit_filename}")

if __name__ == "__main__":
    main()
