
import os
from typing import List, Dict
import logging

from src.utils.ConfigData import ConfigJobGen

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobScriptGenerator:
    def __init__(self, config: ConfigJobGen):
        """
        Initializes the JobScriptGenerator with the given configuration.

        Parameters:
        - config (ConfigJobGen): The configuration for the job script generator.
        """
        self.config = config
        self.generated_scripts = []

    def generate_job_script(self, script_filename: str,
                            jobname: str, relative_path: str, args: str):
        """
        Generates a job script for a batch processing system.

        Parameters:
        - script_filename (str): The name of the script file to be created.
        - jobname (str): The name of the job.
        - relative_path (str): The relative path to the Python module to run.
        - args (str): The arguments to pass to the Python module.
        """
        project_dir = self.config.project_base_dir
        job_script = f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o {project_dir}/log/{jobname}.out
#PBS -e {project_dir}/log/{jobname}.err
#PBS -l nodes={self.config.nodes}:ppn={self.config.ppn},walltime={self.config.walltime}
#PBS -u {self.config.user_name}
#PBS -M {self.config.user_email}
#PBS -m ae
#PBS -q mini

source ~/.bashrc
conda activate {self.config.conda_env}
cd {project_dir}
python -m {relative_path} {args}
"""

        # Ensure the log directory exists
        os.makedirs(os.path.join(project_dir, 'log'), exist_ok=True)

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(script_filename), exist_ok=True)
            with open(script_filename, 'w') as file:
                file.write(job_script)
            self.generated_scripts.append(script_filename)  # Add script filename to the list
            logger.info("Job script created successfully: %s", script_filename)
        except Exception as e:
            logger.error("Failed to create job script: %s", str(e))

    def generate_submission_script(self, jobname: str):
        """
        Generates a submission script to submit all generated job scripts.

        Parameters:
        - jobname (str): The name of the job.
        """
        project_dir = self.config.project_base_dir
        submit_filename = os.path.join(project_dir, "job", "submission", f"submit_{jobname}.sh")

        # Ensure the submission directory exists
        os.makedirs(os.path.dirname(submit_filename), exist_ok=True)

        try:
            with open(submit_filename, 'w') as file:
                for script in self.generated_scripts:
                    file.write(f"qsub {script}\n")
            logger.info("Submission script created successfully: %s", submit_filename)
        except Exception as e:
            logger.error("Failed to create submission script: %s", str(e))

# Example usage:
if __name__ == "__main__":
    config_path = "config.json"
    config = ConfigJobGen.from_json(config_path)
    generator = JobScriptGenerator(config)

    # Example command arguments
    config_id = "example_config"
    source_redshift = "0.5"
    args = f"{config_id} {source_redshift}"

    # Generate job script
    script_filename = "job_script.sh"
    jobname = "smoothed_patch_job"
    relative_path = "src.flatsky.smoothed_patch_flatsky"
    
    generator.generate_job_script(script_filename, jobname, relative_path, args)

    # Generate another job script for demonstration
    script_filename_2 = "job_script_2.sh"
    jobname_2 = "another_job"
    relative_path_2 = "src.another_module"
    args_2 = "arg1 arg2"
    
    generator.generate_job_script(script_filename_2, jobname_2, relative_path_2, args_2)

    # Generate submission script
    generator.generate_submission_script("combined_submission")
