import logging
from pathlib import Path
from lensing_ssc.core.patch.statistics.merger import StatsMerger
from lensing_ssc.utils import PathHandler


def main():
    logging.basicConfig(level=logging.INFO)

    # Find data directories
    data_dirs = PathHandler.find_data_dirs()

    # Merge statistics
    for sl in [5]:  # Processing only smoothing length 5
        for ngal in [0]:  # Noiseless case
            stats_merger = StatsMerger(
                data_dirs=data_dirs,
                sl=sl,
                ngal=ngal,
                opening_angle=10,
                save_dir='/lustre/work/akira.tokiwa/Projects/LensingSSC/output',
                overwrite=True
            )
            stats_merger.run()


if __name__ == "__main__":
    main()