
import json
from glob import glob

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

def main():
    config_file = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    path = config["destination"]
    save_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/img"

    files = glob(path + "/*.npz")

    for file in files:
        data = np.load(file)
        kappa = data["kappa"]
        Nm = data["Nm"]
        nsources = kappa.shape[0]
        zs = float(file.split("_")[-1].split(".")[0])

        
        for i in range(nsources):
            fig = plt.figure(figsize=(10, 5))
            hp.orthview(kappa[i], title="kappa %02.2f" % zs, sub=(1, 2, 1), fig=fig)
            hp.orthview(Nm[i], title="Nm %02.2f" % zs, sub=(1, 2, 2), fig=fig)
            fig.savefig(save_dir + "/kappa_%02.2f_%02.2f.png" % (zs, i))
            plt.close()

if __name__ == "__main__":
    main()