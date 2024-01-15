#! /bin/python

import numpy as np

def write_sbatch(zs):
    zs_list = np.arange(zs, 2.2, 0.1)
    zs_str=''
    for izs in zs_list:
        zs_str+='%.2f '%(izs)
    f = open('jobs/sbatch_zs{0}'.format(zs), 'w')
    content='''#! /bin/bash
#SBATCH --nodes=32
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -C haswell
#SBATCH -t 2:00:00
#SBATCH -J run-wlen-8192-5000-zs{0}
#SBATCH -o zs.{0}_4096-5000.o%j
#SBATCH -e zs.{0}_4096-5000.e%j
#SBATCH --qos=regular
#SBATCH -A mp107
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jialiu@berkeley.edu

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=2

source /global/common/software/m3035/conda-activate.sh 3.7
bcast-pip mpl_aea
bcast-pip mpsort

SRC=/global/cscratch1/sd/akrolew/8192-5000/lightcone/usmesh/
DST=/global/cscratch1/sd/jialiu/fastpm/CorrelatedSims/wlen_jliu
BIN=/global/cscratch1/sd/jialiu/fastpm/CorrelatedSims/wlen_jliu_multiple.py

zs='{1}'
srun python -u $BIN $DST $SRC $zs --zlmin 0.0 --zlmax 2.2 --zstep=0.2 --nside=4096

#BIN=/global/cscratch1/sd/jialiu/fastpm/CorrelatedSims/wlen_jliu_1zs.py
#for zs in `seq {0} 0.1 2.2`
#do echo $zs
#srun python -u $BIN $DST $SRC $zs --zlmin 0.0 --zlmax 2.2 --zstep=0.02 --nside=4096 &
#wait
#done
'''.format(zs, zs_str)
    f.write(content)

for zs in np.arange(1.2, 1.3, 0.01):#(0.1, 0.2, 0.01):
    write_sbatch(zs)


