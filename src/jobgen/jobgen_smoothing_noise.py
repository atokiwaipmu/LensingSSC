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

    noise_files = glob.glob(f"{config_analysis.resultsdir}/noise/noise_*.fits")

    # Generate job script
    relative_path = "src.analysis.map_smoothing"
    for noise_file in noise_files:
        noise_survey = os.path.basename(noise_file).split('_')[1]
        for config_id in ['tiled', 'bigbox']:
            script_dir = os.path.join(job_dir, "scripts", f"smoothing_{config_id}")
            os.makedirs(script_dir, exist_ok=True)
            for z in config_data.zs_list:
                for sl in config_analysis.sl_arcmin:
                    jobname = f"smoothing_{config_id}_{z:.1f}_{sl}_{noise_survey}"
                    script_filename = os.path.join(script_dir, f"job_{jobname}.sh")
                    args = f"{config_id} {z} {sl} --noise_file {noise_file}"
                    generator.generate_job_script(script_filename, jobname, relative_path, args)

    # Generate submission script
    generator.generate_submission_script(f"smoothing_all")


if __name__ == "__main__":
    main()
