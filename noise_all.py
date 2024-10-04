from src.job_generator import JobGenerator
from src.utils import find_data_dirs

def main():
    data_dirs = find_data_dirs()
    for datadir in data_dirs[2:5]:
        job_generator = JobGenerator(datadir, queue="tiny", ppn=1)
        job_generator.gen_script("src.noise_generator", "noisegen", if_omp=False, if_submit=True)

if __name__ == "__main__":
    main()