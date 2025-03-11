#!/bin/bash

# Define parameters
box_types=("tiled" "bigbox")
source_redshifts=(0.5 1.0 1.5 2.0 2.5)

# Create directory for job scripts if it doesn't exist
script_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/job/scripts"
mkdir -p "$script_dir"

# Loop through all combinations and submit jobs
for box_type in "${box_types[@]}"; do
    for zs in "${source_redshifts[@]}"; do
        # Create a uniquely named script file with timestamp
        timestamp=$(date +%Y%m%d_%H%M%S)
        script_file="${script_dir}/patch_${box_type}_${zs}.sh"
        
        cat > "$script_file" << EOF
#!/bin/bash
#PBS -N patch_${box_type}_${zs}
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_${box_type}_${zs}.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_${box_type}_${zs}.err
#PBS -l nodes=1:ppn=52,walltime=100:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q mini

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

python -m lensing_ssc.run.run_patch_analysis --box_type ${box_type} --zs ${zs} --overwrite
EOF
        
        # Make script executable
        chmod +x "$script_file"
        
        # Submit the job
        qsub "$script_file"
        
        echo "Submitted job for box_type=${box_type}, zs=${zs} - Script saved as ${script_file}"
    done
done