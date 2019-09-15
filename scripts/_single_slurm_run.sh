#!/usr/bin/env zsh

### Job name
#SBATCH --job-name=JOB
##SBATCH --account=um_dke

### File for the output
#SBATCH --output=/home/no316758/slurm_results/JOB_OUTPUT.%j.txt

### request one gpu per node
#SBATCH --gres=gpu:volta:1

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH --time=119:59:30

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=16G

### The last part consists of regular shell commands:
### Change to working directory
cd /home/no316758/projects/eeg_thesis

### Execute your application
source ~/anaconda3/bin/activate eeg

### add -u flag so that python print will be unbuffered and therefore show up with bpeek (-f) command
python -um experiment.run
#python -um experiment.bd_bcic