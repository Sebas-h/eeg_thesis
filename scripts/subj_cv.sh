##!/usr/bin/env bash
#!/usr/bin/env zsh

for i in 0 1 2 3 4 5 6 7 8
do
    sbatch --job-name=MYJOB.$i /home/no316758/projects/eeg_thesis/scripts/args_pipeline.sh $i 0
done
