
#!/bin/bash
#PBS -N test
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/test.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/test.err
#PBS -l nodes=4:ppn=52,walltime=24:00:00
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate lssc
cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

MPIRUN=/opt/intel/psxe2018/compilers_and_libraries_2018.5.274/linux/mpi/intel64/bin/mpirun
DATA_DIR=/lustre/work/akira.tokiwa/Projects/LensingSSC/data/bigbox/rfof_proc131072_nc6144_size3750_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s120_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8
CONFIG_FILE=/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_default.yaml
$MPIRUN -np 4 python -m src.patchprocessor_mpi $DATA_DIR $CONFIG_FILE