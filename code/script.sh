#!/usr/bin/env zsh
 
### Job name
#BSUB -J SERIALJOB -P um_dke
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o SERIALJOB.%J.%I
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 1:23
 
### Request memory you need for your job in TOTAL in MB
#BSUB -M 1024
 
### Change to the work directory
cd /home/no316758/projects/eeg_thesis/eeg_classify/my_code
 
### Execute your application
source activate eeg
python mycropped_bcic-iv-2a.py