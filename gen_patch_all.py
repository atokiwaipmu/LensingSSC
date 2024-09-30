
from src.job_generator import JobGenerator
from src.utils import find_data_dirs

def main():
    data_dirs = find_data_dirs()
    for datadir in data_dirs[5:]:
        job_generator = JobGenerator(datadir, queue="mini")
        job_generator.gen_script("src.patch_generator", "gen_patch", if_omp=False, if_submit=True)

if __name__ == "__main__":
    main()