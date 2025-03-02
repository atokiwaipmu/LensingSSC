from dataclasses import dataclass
from pathlib import Path
from typing import Set, ClassVar, Pattern
import re

@dataclass(frozen=True)
class PathConfig:
    """
    Holds default configuration values for path and file handling.
    """
    DEFAULT_WORKDIR: Path = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/")
    DATA_SUBPATH: str = "data/*/*/usmesh"
    KAPPA_DIR: str = "kappa"
    FITS_PATTERN: str = "*.fits"
    MAX_WORKERS: int = 4


@dataclass(frozen=True)
class PathPatterns:
    """
    Regular expression patterns for extracting information from paths.
    """
    SEED: ClassVar[Pattern] = re.compile(r'_s(\d+)')
    REDSHIFT: ClassVar[Pattern] = re.compile(r'_zs(\d+\.\d+)')
    BOX_SIZE: ClassVar[Pattern] = re.compile(r'_size(\d+)')
    OA: ClassVar[Pattern] = re.compile(r'_oa(\d+)')
    SL: ClassVar[Pattern] = re.compile(r'_sl(\d+)')
    NOISE: ClassVar[Pattern] = re.compile(r'_ngal(\d+)')


@dataclass(frozen=True)
class BoxSizes:
    """
    Valid sets of box sizes for distinguishing 'bigbox' from 'tiled'.
    """
    BIG_BOX: Set[int] = frozenset({3750, 5000})
    SMALL_BOX: Set[int] = frozenset({625})