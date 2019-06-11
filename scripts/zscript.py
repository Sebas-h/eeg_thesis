import os
import subprocess
from util import config

config = config.load_cfg(None)

path_script = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'scripts/args_run.sh'))

if config['server']['full_cv']:
    dataset_name = config['experiment']['dataset']
    # n_subjects = [x for x in
    #               range(1, config['data'][dataset_name]['n_subjects'] + 1)]
    # n_folds = [x for x in range(config['experiment']['n_folds'])]
    for subject_id in range(1, 2):
        for i_fold in range(3):
            subprocess.Popen(
                f'sbatch --job-name=TESTJOB '
                f'{path_script} {subject_id} {i_fold}',
                shell=True
            )
else:
    subject_id = config['experiment']['subject_id']
    i_valid_fold = config['experiment']['i_valid_fold']
    subprocess.Popen(
        f'sbatch --job-name=TESTJOB '
        f'{path_script} {subject_id} {i_valid_fold}',
        shell=True
    )
