from src.job_generator import JobGenerator
from src.utils import find_data_dirs

def job_kappa():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="mini", ppn=20)
        job_generator.gen_script("src.kappa_constructor", "kappa", if_omp=False, if_submit=True)

def job_smooth():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="mini", ppn=52)
        job_generator.gen_script("src.kappa_smoother", "smoothing", if_omp=True, if_submit=True)

def job_noise():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="tiny", ppn=1)
        job_generator.gen_script("src.noise_generator", "noisegen", if_omp=False, if_submit=True)

def job_clkk():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="mini", ppn=52)
        job_generator.gen_script("src.clkk_calculator", "clkk", if_omp=True, if_submit=True)

def job_patch():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="mini", ppn=52)
        job_generator.gen_script("src.patch_generator", "patchgen", if_omp=False, if_submit=True)

def job_analysis():
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue="mini", ppn=52)
        job_generator.gen_script("analysis", "analysis", if_omp=False, if_submit=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, help="Job to run")
    args = parser.parse_args()

    if args.job == "kappa":
        job_kappa()
    elif args.job == "smooth":
        job_smooth()
    elif args.job == "noise":
        job_noise()
    elif args.job == "clkk":
        job_clkk()
    elif args.job == "patch":
        job_patch()
    elif args.job == "analysis":
        job_analysis()
    else:
        raise ValueError("Invalid job name")