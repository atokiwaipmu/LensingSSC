
import os
import sys
import numpy as np
import dask.array as da
import healpy as healpix
import gc
import json

import bigfile
import resource
import nbodykit
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
from nbodykit.utils import GatherArray

from mpi4py import MPI
nbodykit.setup_logging()
nbodykit.set_options(dask_chunk_size=1024 * 1024)
nbodykit.set_options(global_cache_size=0)

from src.utils.read_range import read_range
from src.utils.wlen_calc import wlen

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def weighted_map(local_ipix, local_weights, npix):
    # Get the unique indices and the index each value belongs to
    local_unique_indices, local_index_counts = np.unique(local_ipix, return_inverse=True)

    # Count the occurrences of each index using np.bincount
    local_counts = np.bincount(local_index_counts) # len(local_counts) = len(local_unique_indices)
    local_weights = np.bincount(local_index_counts, weights=local_weights)

    pairs = np.empty(len(local_unique_indices) + 1, dtype=[('ipix', 'i4'), ('N', 'i4'), ('weights', 'f8') ])
    pairs['ipix'][:-1] = local_unique_indices
    pairs['weights'][:-1] = local_weights
    pairs['N'][:-1] =local_counts

    pairs['ipix'][-1] = npix - 1 # trick to make sure the final length is correct.
    pairs['weights'][-1] = 0
    pairs['N'][-1] = 0

    w = np.bincount(pairs['ipix'],weights=pairs['weights']) # len(w) = npix
    N = np.bincount(pairs['ipix'],weights=pairs['N'])

    return w, N

class CosmicShearPipeline:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.dst = self.config['destination']
        self.cat = None
        self.z = None
        self.zs_list = self.config['zs']
        self.ds_list = None
        self.nsources = len(self.zs_list)
        self.nside = self.config['nside']
        self.npix = None
        self.nbar = None
        self.inititalize()
        self.Om = self.cat.attrs['OmegaM'][0]

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def create_output_directory(self):
        if not os.path.exists(self.dst):
            os.makedirs(self.dst)

    def initialize_catalog(self):
        path = self.config['source']
        dataset = self.config['dataset']
        self.cat = BigFileCatalog(path, dataset=dataset)

    def calculate_redshift_range(self):
        zlmin = self.config['zlmin']
        zlmax = self.config.get('zlmax', None)
        zstep = self.config['zstep']
        if zlmax is None:
            zlmax = max(self.zs_list)
        Nsteps = int(np.round((zlmax - zlmin) / zstep))
        if Nsteps < 2:
            Nsteps = 2
        self.z = np.linspace(zlmax, zlmin, Nsteps, endpoint=True)

    def calculate_comoving_distance(self):
        self.ds_list = Planck15.comoving_distance(self.zs_list)

    def initialize_healpix(self):
        self.npix = healpix.nside2npix(self.nside)

    def calculate_number_density(self):
        self.nbar = (self.cat.attrs['NC'] ** 3 / self.cat.attrs['BoxSize'] ** 3 * self.cat.attrs['ParticleFraction'])[0]

    def inititalize(self):
        self.create_output_directory()
        self.initialize_catalog()
        self.calculate_redshift_range()
        self.calculate_comoving_distance()
        self.initialize_healpix()
        self.calculate_number_density()

def main(config_file='config.json'):
    pipeline = CosmicShearPipeline(config_file)
    rank = pipeline.cat.comm.rank
    size = pipeline.cat.comm.size
    logger = pipeline.cat.logger

    if rank == 0:
        logger.info("Starting Cosmic Shear Calculation")
        logger.info(f"redshift bins: {pipeline.z}")

    for z2, z1 in zip(pipeline.z[1:][::-1], pipeline.z[:-1][::-1]):
        sliced = read_range(pipeline.cat, 1/(1 + z1), 1 / (1 + z2))
        if sliced.csize == 0: continue
        if rank == 0:
            logger.info("Processing redshift bin %f < z < %f" % (z1, z2))

        # Check if data is already computed
        if rank == 0:
            fname = os.path.join(pipeline.dst,"kappa_%02.2f_%02.2f.npz" % (z1, z2))
            flag = os.path.exists(fname)
        else:
            flag = None
        pipeline.cat.comm.barrier()
        flag = pipeline.cat.comm.bcast(flag, root=0)
        if flag:
            if rank == 0:
                logger.info("Data already exists for redshift bin %f < z < %f" % (z1, z2))
            continue

        start_index = sliced.csize // size * rank
        end_index = sliced.csize // size * (rank + 1)

        tmp_cat = sliced[start_index:end_index].copy()

        dl = (abs(tmp_cat['Position'] ** 2).sum(axis=-1)) ** 0.5 
        ra = tmp_cat['RA']
        dec = tmp_cat['DEC']
        zl = (1 / tmp_cat['Aemit'] - 1)
        dl = dl.persist()
        if rank == 0:
            logger.info("dl persist done")
            logger.info("memory usage by dl: %f GB" % (dl.nbytes / 1024 ** 3))
        area = (4 * np.pi / pipeline.npix) * dl**2

        ipix = da.apply_gufunc(lambda ra, dec, nside: healpix.ang2pix(nside, np.radians(90 - dec), np.radians(ra)),
                            '(),()->()', ra, dec, nside=pipeline.nside)
        local_ipix = ipix.compute()
        if rank == 0:
            logger.info("ipix computed")
            logger.info("memory usage by ipix: %f GB" % (local_ipix.nbytes / 1024 ** 3))

        del ipix
        gc.collect()

        if rank == 0:
            kappa_list = np.zeros([pipeline.nsources, pipeline.npix], dtype="f8")
            Nm_list = np.zeros([pipeline.nsources, pipeline.npix], dtype="i4")
            logger.info("pre-allocation done")
            logger.info("memory usage by kappa_list, Nm_list: %f GB" % ((kappa_list.nbytes + Nm_list.nbytes) / 1024 ** 3))

            dir_name = os.path.join(pipeline.dst, "kappa_%02.2f_%02.2f" % (z1, z2))
            os.makedirs(dir_name, exist_ok=True)
            logger.info("saving directory created at %s" % dir_name)
        else:
            kappa_list = None
            Nm_list = None

        for i, (zs, ds) in enumerate(zip(pipeline.zs_list, pipeline.ds_list)):
            if rank == 0:
                logger.info("Processing source redshift %f" % zs)

            # Check if data is already computed
            if rank == 0:
                fname = os.path.join(dir_name, "kappa_%02.2f.npz" % zs)
                flag = os.path.exists(fname)
            else:
                flag = None
            pipeline.cat.comm.barrier()
            flag = pipeline.cat.comm.bcast(flag, root=0)
            if flag:
                if rank == 0:
                    logger.info("Data already exists for source redshift %f" % zs)
                continue
        
            LensKernel = da.apply_gufunc(lambda dl, zl, Om, ds: wlen(Om, dl, zl, ds), 
                                            "(), ()-> ()",
                                            dl, zl, Om=pipeline.Om, ds=ds)

            weights = (LensKernel / (area * pipeline.nbar))

            local_weights = weights.compute()
            if rank == 0:
                logger.info("local_weights computed")
                logger.info("memory usage by local_weights: %f GB" % (local_weights.nbytes / 1024 ** 3))
            
            del weights, LensKernel
            gc.collect()

            local_w, local_N = weighted_map(local_ipix, local_weights, pipeline.npix)
            if rank == 0:
                logger.info("weighted_map done")
                logger.info("memory usage by w, N: %f GB" % ((local_w.nbytes + local_N.nbytes) / 1024 ** 3))
            pipeline.cat.comm.barrier()

            if rank == 0:
                pipeline.cat.comm.Reduce(MPI.IN_PLACE, local_w, op=MPI.SUM, root=0)
                logger.info("summed up local w for source plane %g" % zs)
                logger.info("memory usage by w: %f GB" % (local_w.nbytes / 1024 ** 3))

                pipeline.cat.comm.Reduce(MPI.IN_PLACE, local_N, op=MPI.SUM, root=0)
                logger.info("summed up local N for source plane %g" % zs)
                logger.info("memory usage by N: %f GB" % (local_N.nbytes / 1024 ** 3))
            else:
                pipeline.cat.comm.Reduce(local_w, None, op=MPI.SUM, root=0)
                pipeline.cat.comm.Reduce(local_N, None, op=MPI.SUM, root=0)

            if rank == 0:
                kappa_list[i] = local_w
                Nm_list[i] = local_N

                fname = os.path.join(dir_name, "kappa_%02.2f.npz" % zs)
                logger.info("saving maps to %s" % fname)
                np.savez(fname, kappa=local_w, Nm=local_N)

            # delete variables in this loop and collect garbage
            del local_w, local_N, local_weights
            gc.collect()


        if rank == 0:
            fname = os.path.join(pipeline.dst,"kappa_%02.2f_%02.2f.npz" % (z1, z2))
            np.savez(fname, kappa=kappa_list, Nm=Nm_list)
            logger.info("all maps saved to %s" % fname)

        del local_ipix
        del dl, area
        del kappa_list, Nm_list
        gc.collect()

    logger.info("Cosmic Shear Calculation Done")

if __name__ == "__main__":
    main("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config.json")