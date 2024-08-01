import glob
import os

from src.utils.jobgen_template import JobScriptGenerator
from src.utils.ConfigData import ConfigJobGen, ConfigData, ConfigAnalysis

def main():
    job_dir = "/lustre/work/akira.tokiwa/Projects/LensingSSC/job"

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    config_jobgen_path = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_jobgen.json')
    config_jobgen = ConfigJobGen.from_json(config_jobgen_path)
    generator = JobScriptGenerator(config_jobgen)

    # Generate job script
    relative_path = "src.pipeline_flatsky"
    for config_id in ['tiled', 'bigbox']:
        script_dir = os.path.join(job_dir, "scripts", f"flatsky_{config_id}")
        os.makedirs(script_dir, exist_ok=True)
        for z in config_data.zs_list:
            for sl in config_analysis.sl_arcmin:
                for survey in ['noiseless', 'Euclid-LSST', 'DES-KiDS', 'HSC', 'Roman']:
                    jobname = f"flatsky_{config_id}_{z:.1f}_{sl}_{survey}"
                    script_filename = os.path.join(script_dir, f"job_{jobname}.sh")
                    args = f"--config_sim {config_id} --zs {z:.1f} --sl {sl} --survey {survey}"
                    generator.generate_job_script(script_filename, jobname, relative_path, args)

    # Generate submission script
    generator.generate_submission_script("flatsky")


if __name__ == "__main__":
    main()
