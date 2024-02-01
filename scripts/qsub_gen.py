#! /bin/python

import argparse

def write_qsub(zlmin, zlmax, zstep, zs_list, nside=8192, datatype='tiled'):
    zs_str=''
    for izs in zs_list:
        zs_str+='%.2f '%(izs)
    f = open('/lustre/work/akira.tokiwa/Projects/LensingSSC/job/scripts/qsub_{0}_zl{1}_{2}'.format(datatype, zlmin, zlmax), 'w')
    if datatype == 'tiled':
        datadir="rfof_proc4096_nc1024_size625_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8_LCDM_10tiled"
    elif datatype == 'bigbox':
        datadir="rfof_proc262144_nc8192_size5000_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8"
    content=f'''#!/bin/bash

#!/bin/bash
#PBS -N wlen_{datatype}_{zlmin}_{zlmax}
#PBS -o /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/{datatype}_{zlmin}_{zlmax}.out
#PBS -e /lustre/work/akira.tokiwa/Projects/LensingSSC/job/log/{datatype}_{zlmin}_{zlmax}.err
#PBS -l nodes=4:ppn=32,walltime=99:99:99
#PBS -u akira.tokiwa
#PBS -M akira.tokiwa@ipmu.jp
#PBS -m ae
#PBS -q small

source ~/.bashrc
conda activate cfastpm

cd /lustre/work/akira.tokiwa/Projects/LensingSSC/

SRC=/lustre/work/akira.tokiwa/globus/fastpm/rfof/{datadir}/usmesh/
DST=/lustre/work/akira.tokiwa/globus/fastpm/rfof/{datadir}/wlen/

zs='{zs_str}'
python -m scripts.wlen $DST $SRC $zs --zlmin {zlmin} --zlmax {zlmax} --zstep={zstep} --nside={nside}
'''
    f.write(content)

if __name__ == '__main__':
    # Usage: python qsub_gen.py 0.0 2.2 0.2 --datatype tiled
    zs_list = [0.5, 1.0, 2.0, 3.0]
    parser = argparse.ArgumentParser()
    parser.add_argument('zlmin', type=float)
    parser.add_argument('zlmax', type=float)
    parser.add_argument('zstep', type=float)
    #parser.add_argument('zs_list', nargs='+', type=float)
    parser.add_argument('--datatype', type=str, default='tiled')
    parser.add_argument('--nside', type=int, default=8192)
    args = parser.parse_args()
    write_qsub(args.zlmin, args.zlmax, args.zstep, zs_list, args.nside, args.datatype)