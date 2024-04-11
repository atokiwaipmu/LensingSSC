import os
import gc
import numpy as np
import json
import bigfile
import nbodykit
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
from glob import glob

def load_and_process_file(file):
    data = np.load(file)
    return data["kappa"].astype(np.float64), data["Nm"]

def main():
    config_file = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    path = config["destination"]
    cat = BigFileCatalog(config['source'], dataset=config['dataset'])
    nbar = (cat.attrs['NC'] ** 3 / cat.attrs['BoxSize'] ** 3 * cat.attrs['ParticleFraction'])[0]
    zs_list = config["zs"]
    ds_list = Planck15.comoving_distance(zs_list)

    file_list = []
    for zs in zs_list:
        file_pattern = f"{path}/*/kappa_{zs:02.2f}.npz"
        files = sorted(glob(file_pattern))
        file_list.append(files)

    for i, (zs, ds) in enumerate(zip(zs_list, ds_list)):
        kappa = np.zeros(12 * config["nside"] ** 2).astype(np.float64)
        Nm = np.zeros(12 * config["nside"] ** 2).astype(np.int32)
        for j in range(len(file_list[i])):
            file = file_list[i][j]
            data = np.load(file)
            tmp_Nm = data["Nm"]
            print(f"Source plane at {zs} from {file} has {np.sum(tmp_Nm)} galaxies.")
            kappa += data["kappa"].astype(np.float64)
            Nm += tmp_Nm.astype(np.int32)

        fname = f"{path}/WL-{zs:02.2f}-N{config['nside']:04d}"
        with bigfile.File(fname, create=True) as ff:
            ds1 = ff.create_from_array("kappa", kappa, Nfile=1)
            ds2 = ff.create_from_array("Nm", Nm, Nfile=1)
            for d in ds1, ds2:
                d.attrs['nside'] = config["nside"]
                d.attrs['zlmin'] = config["zlmin"]
                d.attrs['zlmax'] = config["zlmax"]
                d.attrs['zs'] = zs
                d.attrs['ds'] = ds
                d.attrs['nbar'] = nbar
            print(f"Source plane at {zs} written.")

        del kappa, Nm
        gc.collect()

if __name__ == "__main__":
    nbodykit.setup_logging()
    nbodykit.set_options(dask_chunk_size=1024 * 1024)
    nbodykit.set_options(global_cache_size=0)
    main()