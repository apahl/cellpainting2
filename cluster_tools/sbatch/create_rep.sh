#!/bin/bash -l

#SBATCH --job-name=createrp
#SBATCH --array=1-20
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/createrp_%A-%a.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/createrp_%A-%a.txt
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=4096
# #SBATCH --time=24:00:00

ORIG_DIR=$(pwd)

source activate chem
sleep 5

if [[ $SLURM_ARRAY_TASK_ID == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CreateRp started..."
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CreateRp started..." >> $ORIG_DIR/job_info.txt
  # remove existing report dir, start fresh
  rm -rf /ptmp/apahl/cp/profiles/reports
else
  sleep 120
fi

PLATES=$(get_plates)
create_reports -p $PLATES -t $SLURM_ARRAY_TASK_ID -n 10
source deactivate

if [[ $SLURM_ARRAY_TASK_ID == 10 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CreateRp finished." >> $ORIG_DIR/job_info.txt
fi
