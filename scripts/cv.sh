##!/usr/bin/env bash
#!/usr/bin/env zsh

for i in 0 1 2 3 4 5 6 7 8
#for i in 0
do
    for j in 0 1 2 3
#    for j in 0
    do
        sbatch --job-name=MYJOB.$i.$j /home/no316758/projects/eeg_thesis/scripts/args_pipeline.sh $i $j
    done
done
