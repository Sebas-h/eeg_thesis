#!/usr/bin/env zsh

i=""
for j in 0 1 2 3
do
    sbatch --job-name=JOB.${j} /home/no316758/projects/eeg_thesis/scripts/_args_run.sh ${i} ${j}
done
