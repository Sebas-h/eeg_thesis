##!/usr/bin/env bash
#!/usr/bin/env zsh

for i in 0
do
    for j in 0 1
    do
        sbatch /home/no316758/projects/eeg_thesis/scripts/args_pipeline.sh $i $j
    done
done
