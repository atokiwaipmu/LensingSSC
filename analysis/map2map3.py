import numpy as np
import os,sys
from analysis.HOS_Y1.HOScodes import *
from glob import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(dir_results, filenames, nside=8192, thetas=[4,8,16,32], dataformat='npy'):
    logging.info("Starting the kappa maps processing.")
    kappa_maps = kappacodes(dir_results=dir_results, filenames=filenames, nside=nside)
    
    # Depending on the format, different reading functions are invoked
    if dataformat == 'fits':
        kappa_maps.readmaps_healpy()
    elif dataformat == 'npy':
        kappa_maps.readmaps_npy()
       
    for i, map_i in enumerate(kappa_maps.mapbins):
        logging.info(f'Tomo bin {i}')
        kappa_maps.run_map2alm(i)
        logging.info('Map2 transformation done.')
        kappa_maps.run_map3(i, thetas=thetas)
        logging.info('Map3 processing done.')
        
        logging.info('Starting cross correlation...')
        for j in range(i):
            kappa_maps.run_map2alm(Nmap1=i, Nmap2=j, is_cross=True)
            logging.info(f'Map2 cross {i} {j} done.')

if __name__ == '__main__':
    config = sys.argv[1]
    dataformat = sys.argv[2]

    dir_results = f"/lustre/work/akira.tokiwa/Projects/LensingSSC/results/{config}/"
    filenames = glob(f"/lustre/work/akira.tokiwa/Projects/LensingSSC/results/{config}/wlen/*/*.{dataformat}")
    print(filenames)
    main(dir_results, filenames)
    logging.info("All done.")