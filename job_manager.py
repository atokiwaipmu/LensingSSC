from typing import Dict
from src.job_generator import JobGenerator
from src.utils import find_data_dirs

def run_job(
    module_name: str,
    script_name: str,
    queue: str,
    ppn: int,
    if_omp: bool,
    if_submit: bool,
) -> None:
    """
    Generates and submits jobs based on the provided configuration.

    Parameters:
    - module_name (str): The module to run.
    - script_name (str): The name of the script.
    - queue (str): The queue to submit the job to.
    - ppn (int): Number of processors per node.
    - if_omp (bool): Whether to use OpenMP.
    - if_submit (bool): Whether to submit the job after generation.
    """
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        job_generator = JobGenerator(datadir, queue=queue, ppn=ppn)
        job_generator.gen_script(
            module_name,
            script_name,
            if_omp=if_omp,
            if_submit=if_submit,
        )

if __name__ == "__main__":
    import argparse
    import sys

    # Define job configurations
    job_configs: Dict[str, Dict[str, any]] = {
        "kappa": {
            "module_name": "src.kappa_constructor",
            "script_name": "kappa",
            "queue": "mini",
            "ppn": 52,
            "if_omp": False,
            "if_submit": True,
        },
        "smooth": {
            "module_name": "src.kappa_smoother",
            "script_name": "smoothing",
            "queue": "mini",
            "ppn": 52,
            "if_omp": True,
            "if_submit": True,
        },
        "noise": {
            "module_name": "src.noise_generator",
            "script_name": "noisegen",
            "queue": "mini",
            "ppn": 1,
            "if_omp": False,
            "if_submit": True,
        },
        "clkk": {
            "module_name": "src.clkk_calculator",
            "script_name": "clkk",
            "queue": "mini",
            "ppn": 52,
            "if_omp": True,
            "if_submit": True,
        },
        "patch": {
            "module_name": "src.patch_generator",
            "script_name": "patchgen",
            "queue": "mini",
            "ppn": 52,
            "if_omp": False,
            "if_submit": True,
        },
        "analysis": {
            "module_name": "analysis",
            "script_name": "analysis",
            "queue": "mini",
            "ppn": 52,
            "if_omp": False,
            "if_submit": True,
        },
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Job submission script")
    parser.add_argument("job", type=str, help="Job to run")
    args = parser.parse_args()

    job_name = args.job.lower()

    # Run the specified job
    if job_name in job_configs:
        config = job_configs[job_name]
        run_job(**config)
    else:
        valid_jobs = ", ".join(job_configs.keys())
        print(f"Error: Invalid job name '{job_name}'. Valid options are: {valid_jobs}")
        sys.exit(1)