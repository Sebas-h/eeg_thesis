import subprocess

path = '/home/no316758/projects/eeg_thesis/scripts/args_run.sh'
subprocess.Popen([f'{path} {1} {0}'], shell=True)
subprocess.Popen([f'{path} {1} {1}'], shell=True)
