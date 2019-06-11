import subprocess
import os

path = '/home/no316758/projects/eeg_thesis/scripts/args_run.sh'
# path = '/Users/sebas/code/thesis/scripts/args_run.sh'
subprocess.Popen(f'{path} {1} {0}', shell=True,
                 stdin=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL)
subprocess.Popen(f'{path} {1} {1}', shell=True,
                 stdin=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL)

# subprocess.Popen([f'{path} {1} {0}'])
# subprocess.Popen([f'{path} {1} {1}'])

# args = "#!/usr/bin/env zsh\nfor i in 1 2 3 4 5 6 7 8 9\ndo" \
#        "\nfor j in 0 1 2 3\ndo\nsbatch --job-name=JOB.${i}.${j} " \
#        "/home/no316758/projects/eeg_thesis/scripts/args_run.sh " \
#        "${i} ${j}\ndone\ndone"
#
# subprocess.Popen(args, shell=True)
