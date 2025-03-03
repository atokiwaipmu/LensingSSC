#!/bin/bash

# Define parameters
box_types=("tiled" "bigbox")
source_redshifts=(0.5 1.0 1.5 2.5)

# Loop through all combinations and submit jobs
for box_type in "${box_types[@]}"; do
    for zs in "${source_redshifts[@]}"; do
        # Create a temporary script file with the parameters substituted
        temp_script=$(mktemp)
        
        cat > "$temp_script" << EOF
#!/bin/bash
#PBS -N patch_${box_type}_${zs}
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_${box_type}_${zs}.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_${box_type}_${zs}.err
#PBS -l nodes=1:ppn=1,walltime=100:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q tiny

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

python -m lensing_ssc.run.run_patch_analysis --box_type ${box_type} --zs ${zs} --overwrite
EOF
        
        # Submit the job
        qsub "$temp_script"
        
        # Remove the temporary file
        rm "$temp_script"
        
        echo "Submitted job for box_type=${box_type}, zs=${zs}"
    done
done