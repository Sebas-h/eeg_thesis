import subprocess

path = '/home/no316758/projects/eeg_thesis/scripts/args_run.sh'
# path = '/Users/sebas/code/thesis/scripts/args_run.sh'

subprocess.Popen(f'sbatch --job-name=TESTJOB {path} {1} {0}', shell=True)
subprocess.Popen(f'sbatch --job-name=TESTJOB {path} {1} {1}', shell=True)
subprocess.Popen(f'sbatch --job-name=TESTJOB {path} {1} {2}', shell=True)

