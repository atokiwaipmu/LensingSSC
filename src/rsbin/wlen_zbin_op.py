
import os
import numpy as np
import dask.array as da
import healpy as healpix
import gc
import json
import argparse

import nbodykit
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15

from mpi4py import MPI
nbodykit.setup_logging()
nbodykit.set_options(dask_chunk_size=1024 * 1024)
nbodykit.set_options(global_cache_size=0)

from src.utils.read_range import read_range
from src.utils.wlen_calc import wlen

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def weighted_map(ipix, weights, npix):
    # Get the unique indices and the index each value belongs to
    local_unique_indices, local_index_counts = np.unique(ipix, return_inverse=True)

    # Count the occurrences of each index using np.bincount
    local_counts = np.bincount(local_index_counts) # len(local_counts) = len(local_unique_indices)
    local_weights = np.bincount(local_index_counts, weights=weights)

    pairs = np.empty(len(local_unique_indices) + 1, dtype=[('ipix', 'i4'), ('N', 'i4'), ('weights', 'f8') ])
    pairs['ipix'][:-1] = local_unique_indices
    pairs['weights'][:-1] = local_weights
    pairs['N'][:-1] =local_counts

    pairs['ipix'][-1] = npix - 1 # trick to make sure the final length is correct.
    pairs['weights'][-1] = 0
    pairs['N'][-1] = 0

    w = np.bincount(pairs['ipix'],weights=pairs['weights']) # len(w) = npix
    N = np.bincount(pairs['ipix'],weights=pairs['N'])

    del pairs, local_unique_indices, local_index_counts, local_counts, local_weights; gc.collect()

    return w.astype('f8'), N.astype('i4')

class CosmicShearPipeline:
    """A pipeline class for processing cosmic shear data."""

    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.dst = self.config['destination']
        self.cat = BigFileCatalog(self.config['source'], dataset=self.config['dataset'])
        self.z = None
        self.zs_list = self.config['zs']
        self.ds_list = Planck15.comoving_distance(self.zs_list)
        self.nsources = len(self.zs_list)
        self.nside = self.config['nside']
        self.npix = healpix.nside2npix(self.nside)
        self.nbar = (self.cat.attrs['NC'] ** 3 / self.cat.attrs['BoxSize'] ** 3 * self.cat.attrs['ParticleFraction'])[0]
        self.initialize()
        self.Om = self.cat.attrs['OmegaM'][0]

    def validate_config(self, config):
        """Validate required keys in the configuration dictionary."""
        required_keys = ['destination', 'source', 'dataset', 'zs', 'nside', 'zlmin', 'zstep']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing configuration keys: {', '.join(missing_keys)}")

    def load_config(self, config_file: str) -> dict:
        """Load the configuration from a JSON file."""
        with open(config_file, 'r') as file:
            config = json.load(file)
        self.validate_config(config)
        return config

    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.dst):
            os.makedirs(self.dst)

    def calculate_redshift_range(self):
        """Calculate the redshift range array."""
        zlmin = self.config['zlmin']
        zlmax = self.config.get('zlmax', max(self.zs_list))
        zstep = self.config['zstep']
        Nsteps = int(np.round((zlmax - zlmin) / zstep))
        if Nsteps < 2:
            Nsteps = 2
        self.z = np.linspace(zlmax, zlmin, Nsteps, endpoint=True)

    def initialize(self):
        """Initialize all components for the pipeline."""
        self.create_output_directory()
        self.calculate_redshift_range()

def data_preparation(pipeline, z1, z2):
    comm = pipeline.cat.comm
    logger = pipeline.cat.logger

    # Log start and bin information only for the root process
    if comm.rank == 0:
        logger.info("Starting Cosmic Shear Calculation")
        logger.info(f"Redshift bin: {z2:.2f} < z < {z1:.2f}")

    # Calculate the scaling factors and read the range
    scaling_factor_start = 1 / (1 + z1)
    scaling_factor_end = 1 / (1 + z2)
    sliced = read_range(pipeline.cat, scaling_factor_start, scaling_factor_end)

    # Calculate the start and end indices for the current rank
    start_index = sliced.csize // comm.size * comm.rank
    end_index = sliced.csize // comm.size * (comm.rank + 1) if comm.rank != comm.size - 1 else sliced.csize
    cat = sliced[start_index:end_index].copy()
    
    return cat, comm, logger

def process_data(cat, npix, nside):
    """
    Process data for cosmic shear calculations.
    
    Parameters:
    - tmp_cat : The temporary catalog with positions and other necessary attributes.
    - rank : The rank of the process in a distributed setting.
    - logger : Logger for logging information.
    - npix : Number of pixels in the Healpix distribution.
    - nside : The nside parameter for Healpix calculations.
    """
    # Calculate the comoving distance
    dl = (np.abs(cat['Position'] ** 2).sum(axis=-1)) ** 0.5
    ra = cat['RA']
    dec = cat['DEC']
    zl = (1 / cat['Aemit'] - 1)

    # Calculate the area per pixel
    area = (4 * np.pi / npix) * dl**2
    
    # Convert angular coordinates to pixel indices using a gufunc
    ipix = da.apply_gufunc(lambda ra, dec, nside: healpix.ang2pix(nside, np.radians(90 - dec), np.radians(ra)),
                           '(),()->()', ra, dec, nside=nside)
    
    return dl, zl, area, ipix

def prepare_data_storage(comm, pipeline, z1, z2, logger):
    if comm.rank == 0:
        dir_name = os.path.join(pipeline.dst, "kappa_%02.2f_%02.2f" % (z1, z2))
        os.makedirs(dir_name, exist_ok=True)
        logger.info("saving directory created at %s" % dir_name)
    else:
        dir_name = None
    return dir_name

def preallocate_data_storage(comm, pipeline, logger):
    if comm.rank == 0:
        kappa_list = np.zeros([pipeline.nsources, pipeline.npix], dtype="f8")
        Nm_list = np.zeros([pipeline.nsources, pipeline.npix], dtype="i4")
        logger.info("pre-allocation done")
        logger.info("memory usage by kappa_list, Nm_list: %f GB" % ((kappa_list.nbytes + Nm_list.nbytes) / 1024 ** 3))
    else:
        kappa_list = None
        Nm_list = None
    return kappa_list, Nm_list

def check_and_process_data(comm, logger, dir_name):
    if comm.rank == 0:
        # Construct the filename for the current redshift
        fname = os.path.join(dir_name, f"kappa.npz")
        file_exists = os.path.exists(fname)
        if file_exists:
            logger.info(f"Data already exists , skipping.")
        else:
            logger.info(f"No data found, processing needed.")
    else:
        file_exists = None

    # Broadcast the file existence check to all processes
    file_exists = comm.bcast(file_exists, root=0)

    return file_exists

def log_memory(data, data_description, comm, logger):
    # Logging is performed only by the root process
    if (comm.rank == 0):
        logger.info(f"{data_description} done. Memory usage:{data_description}: {data.nbytes / 1024 ** 3:.2f} GB")

def main(config_file, z1=None, z2=None):
    pipeline = CosmicShearPipeline(config_file)
    cat, comm, logger = data_preparation(pipeline, z1, z2)
    
    dir_name = prepare_data_storage(comm, pipeline, z1, z2, logger)
    file_exists = check_and_process_data(comm, logger, dir_name)
    if file_exists:
        return

    batch_size = 10**7
    indices = np.arange(cat.csize//batch_size + 1) * batch_size
    if indices[-1] > cat.csize:
        indices[-1] = cat.csize
    if comm.rank == 0:
        logger.info("Data size: %d, separating into %d batches" % (cat.csize, len(indices)))

    kappa = np.zeros([pipeline.nsources, pipeline.npix], dtype="f8"); log_memory(kappa, "kappa", comm, logger)
    Nm = np.zeros([pipeline.nsources, pipeline.npix], dtype="i4"); log_memory(Nm, "Nm", comm, logger)
    for k in range(len(indices)-1):
        if comm.rank == 0:
            logger.info("Processing batch %d, indices %d to %d" % (k, indices[k], indices[k+1]))
        tmp_cat = cat[indices[k]:indices[k+1]].copy()
        dl, zl, area, ipix = process_data(tmp_cat, pipeline.npix, pipeline.nside)
        dl = dl.persist(); #log_memory(dl, "dl", comm, logger)
        ipix = ipix.compute(); #log_memory(ipix, "ipix", comm, logger)

        for i, (zs, ds) in enumerate(zip(pipeline.zs_list, pipeline.ds_list)):
            #if comm.rank == 0: logger.info("Processing source redshift %f" % zs)

            LensKernel = da.apply_gufunc(lambda dl, zl, Om, ds: wlen(Om, dl, zl, ds), 
                                            "(), ()-> ()",
                                            dl, zl, Om=tmp_cat.attrs['OmegaM'][0], ds=ds)
            weights = (LensKernel / (area * pipeline.nbar))
            weights = weights.compute(); log_memory(weights, "weights", comm, logger)
            del LensKernel; gc.collect()

            tmp_w, tmp_N = weighted_map(ipix, weights, pipeline.npix); #log_memory(tmp_w, "w", comm, logger); log_memory(tmp_N, "N", comm, logger)
            kappa[i] += tmp_w
            Nm[i] += tmp_N

            del tmp_w, tmp_N, weights; gc.collect()

        del dl, zl, area, ipix, tmp_cat; gc.collect()

    logger.info("Rank %d done" % comm.rank)
    comm.barrier()

    if comm.rank == 0:
        # Root process uses MPI.IN_PLACE
        comm.Reduce(MPI.IN_PLACE, kappa, op=MPI.SUM, root=0); log_memory(kappa, "kappa", comm, logger)
        comm.Reduce(MPI.IN_PLACE, Nm, op=MPI.SUM, root=0); log_memory(Nm, "Nm", comm, logger)
    else:
        # Non-root processes send their data
        comm.Reduce(kappa, None, op=MPI.SUM, root=0)
        comm.Reduce(Nm, None, op=MPI.SUM, root=0)

    if comm.rank == 0:
        fname = os.path.join(dir_name, "kappa.npz")
        logger.info("saving combined maps to %s" % fname)
        np.savez(fname, kappa=kappa, Nm=Nm)
        logger.info("Cosmic Shear Calculation Done")
    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str, help="Path to the configuration file")
    args.add_argument("z1", type=float, help="Lower redshift bin")
    args.add_argument("z2", type=float, help="Upper redshift bin")
    args = args.parse_args()

    main(args.config_file, args.z1, args.z2)