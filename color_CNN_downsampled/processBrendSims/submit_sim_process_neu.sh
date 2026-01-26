#!/bin/bash

#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=200:00:00,h_data=50G,highp   #,gpu,RTX2080Ti
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Notify when
#$ -m a

# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate base
conda activate tf_A100_clean


# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

INPUT_FOLDER="/u/home/b/baeria/project-ngarud/hmp_SLiMulations/dann_slimulations_12080244/neutral/"

#run code
python processSLiMsims_ManagerScript.py "neutral_color.npy" 154 200 "$INPUT_FOLDER"


# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "