#!/bin/bash
#PBS -N patch_generator
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_generator.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/patch_generator.err
#PBS -l nodes=1:ppn=1,walltime=100:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q mini

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

python -m lensing_ssc.run.run_patch_generation