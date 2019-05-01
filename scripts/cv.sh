##!/usr/bin/env bash
#!/usr/bin/env zsh

for i in 0 1 2 3 4 5 6 7 8
do
    for j in 0 1 2 3 4
    do
        sbatch /home/no316758/projects/eeg_thesis/scripts/args_pipeline.sh $i $j
    done
done
