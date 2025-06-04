from .base import Provider
from .factory import register_provider
from ..base.data_structures import MapData
import numpy as np
from typing import Tuple, Union
from pathlib import Path

@register_provider("healpix")
class HealpixProvider(Provider):
    def __init__(self):
        super().__init__()
        self._healpy = None
    
    @property
    def name(self) -> str:
        return "healpix"
    
    def _check_dependencies(self):
        try:
            import healpy as hp
            self._healpy = hp
        except ImportError:
            raise ImportError("healpy required for HealpixProvider")
    
    def read_map(self, path: Union[str, Path], **kwargs) -> MapData:
        self.initialize()
        data = self._healpy.read_map(str(path), **kwargs)
        return MapData(data=data, metadata={"source": str(path)})
    
    def write_map(self, map_data: MapData, path: Union[str, Path], **kwargs):
        self.initialize()
        self._healpy.write_map(str(path), map_data.data, **kwargs)
    
    # Include other essential methods: gnomonic_projection, query_polygon, etc. 