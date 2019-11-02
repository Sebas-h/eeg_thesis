import os
import subprocess
import datetime
from util.config import load_cfg
import yaml


def main():
    # get config
    config = load_cfg(None)
    # get vars from config
    dataset_name = config["experiment"]["dataset"]
    dataset_subject_count = config["data"][dataset_name]["n_subjects"]
    experiment_type = config["experiment"]["type"]
    experiment_n_folds = config["experiment"]["n_folds"]
    model_name = config["model"]["name"]
    # path to save csv
    now = (
        str(datetime.datetime.now())
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    results_path = f"/home/no316758/results/{now}_{experiment_type}_{model_name}_{dataset_name}_{experiment_n_folds}/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # copy config
    exp_cfg_path = f"{results_path}config.yaml"
    with open(exp_cfg_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    script_name = "_single_slurm_run.sh"
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), script_name))
    p = subprocess.Popen(
        f"sbatch -o {results_path}sbatch_stdout_%j.txt {script_path} {results_path}",
        shell=True,
    )
    p.wait()


if __name__ == "__main__":
    main()


# path_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '_args_run.sh'))
# subject_id = 1
# i_fold = 0
# p = subprocess.Popen(
#     f'sbatch --job-name=CVJOB {path_script} {subject_id} {i_fold}',
#     shell=True
# )
# p.wait()
# return

# if config['server']['full_cv']:
#     dataset_name = config['experiment']['dataset']
#     n_subjects = [x for x in
#                   range(1,
#                   config['data'][dataset_name]['n_subjects'] + 1)]
#     n_folds = [x for x in range(config['experiment']['n_folds'])]
#     for subject_id in n_subjects:
#         for i_fold in n_folds:
#             subprocess.Popen(
#                 f'sbatch --job-name=CVJOB '
#                 f'{path_script} {subject_id} {i_fold}',
#                 shell=True
#             )
# else:
#     subject_id = config['experiment']['subject_id']
#     i_valid_fold = config['experiment']['i_valid_fold']
#     subprocess.Popen(
#         f'sbatch --job-name=THEJOB '
#         f'{path_script} {subject_id} {i_valid_fold}',
#         shell=True
#     )
