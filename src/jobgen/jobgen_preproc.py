import os
import sys

config = sys.argv[1]

# Configuration
start_index = 20
end_index = 99
output_dir = f'/lustre/work/akira.tokiwa/Projects/LensingSSC/job/scripts/config_{config}'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

job_template = """
#!/bin/bash
#PBS -N wlen_{config}_config_{index}
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/preproc_{config}/wlen_{config}_config_{index}.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/preproc_{config}/wlen_{config}_config_{index}.err
#PBS -l nodes=1:ppn=1,walltime=02:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/
python -m src.masssheet.preproc {config} {index}
"""

for index in range(start_index, end_index + 1):
    job_script_content = job_template.format(index=index, config=config)
    script_path = os.path.join(output_dir, f'job_{index}.sh')
    
    with open(script_path, 'w') as f:
        f.write(job_script_content)
    print(f'Generated job script at {script_path}')
