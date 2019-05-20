#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=MYJOB

### File for the output
#SBATCH --output=/home/no316758/slurm_results/MYJOB_OUTPUT.%j

# set the number of nodes
###SBATCH --nodes=1

# request one gpu per node
#SBATCH --gres=gpu:volta:1

# set the number of nodes
###SBATCH --partition=c16m

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH --time=99:15:30

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=16G

### The last part consists of regular shell commands:
### Change to working directory
cd /home/no316758/projects/eeg_thesis

### Execute your application
source ~/anaconda3/bin/activate eeg
### add -u flag so that python print will be unbuffered and therefore show up with bpeek (-f) command
python -um src.pipeline.main