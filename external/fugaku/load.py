import pathlib
from pathlib import Path
import ctypes
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

class ReprStructure(ctypes.Structure):
    def __repr__(self) -> str:
        values = ", ".join(f"{name}={value}" for name, value in self._asdict().items())
        return f"<{self.__class__.__name__}: {values}>"

    def _asdict(self) -> dict:
        return {field[0]: getattr(self, field[0]) for field in self._fields_}

class HHeader(ReprStructure):
    _pack_ = 1
    _fields_ = (
        ('max_cd_id', ctypes.c_int32),
        ('add_count', ctypes.c_int32),
        ('cd_idx', ctypes.c_int32),
        ('rmin', ctypes.c_double),  # // cd_min in cd_idx
        ('rmax', ctypes.c_double),  # // cd_max in cd_idx

        ('cd_min', ctypes.c_double),  # // global cd_min
        ('cd_max', ctypes.c_double),  # // global cd_max
        ('cd_delta', ctypes.c_double),
        ('ptcl_mass', ctypes.c_double),
        ('pix_volume', ctypes.c_double),

        ('order', ctypes.c_int32),
        ('nside', ctypes.c_int32),
        ('scheme', ctypes.c_int32),
        ('pixel_byte', ctypes.c_int32),
        ('npix', ctypes.c_int64),
    )

class HealpixData:
    def read_hpix_data(self, input_prefix, cd_id):
        filename = f"{input_prefix:s}.{cd_id:d}"
        print("load ", filename)
        header = HHeader()
        read_file = open(filename, "rb")
        read_file.readinto(header)
        if header.pixel_byte == 4: dytpe = np.int32
        if header.pixel_byte == 2: dytpe = np.uint16
        if header.pixel_byte == 1: dytpe = np.uint8

        hpix = np.fromfile(read_file, dtype=dytpe, count=header.npix)
        read_file.close()

        hpix = hpix.astype(np.int32)

        if pathlib.Path(filename + ".1").is_file():
            print("load ", filename + ".1")
            header = HHeader()
            read_file = open(filename, "rb")
            read_file.readinto(header)
            if header.pixel_byte == 4: dytpe = np.int32
            if header.pixel_byte == 2: dytpe = np.uint16
            if header.pixel_byte == 1: dytpe = np.uint8

            _hpix = np.fromfile(read_file, dtype=dytpe, count=header.npix)
            read_file.close()
            hpix += _hpix

        self.header = header
        self.hpix = hpix

@dataclass
class HpixConfig:
    real: int
    boxsize: int # bozsize: [125, 250, 500, 1000, 2000, 4000]
    rmin: float
    rmax: float
    savedir: Path = Path("/vol0503/data/hp230202/u11878/work/LCReplication")
    datadir: Path = Path("/vol0006/mdt1/data/hp230202/u10067/work/LC_multi_box/work")
    delta_r: float = 20.0

class HealpixLoader:
    def __init__(self, config: HpixConfig):
        self.config = config
        self.input_prefix = (self.config.datadir / 
                           f"real{config.real}" / 
                           f"l{config.boxsize}_n{config.boxsize}" / 
                           "hpix_out")

    def _load_hpix_data(self, idx: int) -> Tuple[np.ndarray, float, float]:
        """単一のHealpixデータを読み込む"""
        hdata = HealpixData()
        hdata.read_hpix_data(self.input_prefix, idx)
        return hdata.hpix, hdata.header.rmin, hdata.header.rmax

    def load_and_combine(self) -> Tuple[np.ndarray, float, float]:
        """指定された範囲のHealpixデータを読み込んで結合"""
        idx_min = int(self.config.rmin // self.config.delta_r)
        idx_max = int(self.config.rmax // self.config.delta_r)
        
        hpix_total = 0
        r_min, r_max = float('inf'), float('-inf')
        
        try:
            for idx in range(idx_min, idx_max):
                hpix, rmin, rmax = self._load_hpix_data(idx)
                hpix_total += hpix
                r_min = min(r_min, rmin)
                r_max = max(r_max, rmax)
            
            return hpix_total, r_min, r_max
        
        except Exception as e:
            raise RuntimeError(f"Failed to load Healpix data: {e}")

    def plot_and_save(self, hpix_data: np.ndarray, rmin: float, rmax: float) -> None:
        """Healpixデータをプロットして保存"""
        try:
            fig = plt.figure(figsize=(10, 5))
            title = f"real{self.config.real} l{self.config.boxsize}_n{self.config.boxsize} {rmin:.1f}-{rmax:.1f} Mpc/h"
            hp.mollview(hpix_data, title=title, fig=fig.number)
            
            output_path = (self.config.savedir / 
                         f"real{self.config.real}_l{self.config.boxsize}_n{self.config.boxsize}_{rmin:.1f}-{rmax:.1f}.png")
            fig.savefig(output_path)
            plt.close(fig)
            
        except Exception as e:
            raise RuntimeError(f"Failed to plot and save Healpix data: {e}")

def load_hpix(real: int, boxsize: int, rmin: float, rmax: float, 
              savedir: str = "/vol0503/data/hp230202/u11878/work/LCReplication",
              datadir: str = "/vol0006/mdt1/data/hp230202/u10067/work/LC_multi_box/work") -> None:
    """
    Healpixデータを読み込み、プロットして保存します。

    Args:
        real: リアライゼーションのインデックス
        boxsize: ボックスサイズ
        rmin: 最小距離 (Mpc/h)
        rmax: 最大距離 (Mpc/h)
        savedir: 出力ディレクトリのパス
        datadir: 入力データディレクトリのパス
    """
    config = HpixConfig(real=real, boxsize=boxsize, rmin=rmin, rmax=rmax,
                       savedir=Path(savedir), datadir=Path(datadir))
    
    loader = HealpixLoader(config)
    hpix_data, r_min, r_max = loader.load_and_combine()
    loader.plot_and_save(hpix_data, r_min, r_max)

if __name__ == "__main__":
    load_hpix(real=1, boxsize=125, rmin=1000, rmax=1100)
    load_hpix(real=1, boxsize=250, rmin=1000, rmax=1100)
    load_hpix(real=1, boxsize=500, rmin=1000, rmax=1100)
    load_hpix(real=1, boxsize=1000, rmin=1000, rmax=1100)
    load_hpix(real=1, boxsize=2000, rmin=1000, rmax=1100)
    load_hpix(real=1, boxsize=4000, rmin=1000, rmax=1100)
    