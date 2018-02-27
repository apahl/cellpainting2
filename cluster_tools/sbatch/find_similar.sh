#!/bin/bash -l

#SBATCH --job-name=findsim
#SBATCH --array=1-20  # ALSO CHANGE BELOW IN TWO POSITIONS !!!
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/findsim_%A-%a.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/findsim_%A-%a.txt
#SBATCH --partition=short # small
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=5120
# #SBATCH --time=24:00:00

ORIG_DIR=$(pwd)

source activate chem
sleep 5

if [[ $SLURM_ARRAY_TASK_ID == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: FindSim started..."
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: FindSim started..." >> $ORIG_DIR/job_info.txt
  # remove tmp dir, start fresh
  if [ -d /ptmp/apahl/cp/profiles/data/tmp ]; then
    rm -rf /ptmp/apahl/cp/profiles/data/tmp
  fi
  if [ -d /ptmp/apahl/cp/profiles/data/plots ]; then
    rm -rf /ptmp/apahl/cp/profiles/data/plots
  fi
  if [ -d /ptmp/apahl/cp/profiles/reports ]; then
    rm -rf /ptmp/apahl/cp/profiles/reports
  fi
  rm -rf /ptmp/apahl/cp/profiles/data/sim_refs-*.tsv
else
  sleep 60
fi

PLATES=$(get_plates)
find_similar -p $PLATES -t $SLURM_ARRAY_TASK_ID -n 20
source deactivate

if [[ $SLURM_ARRAY_TASK_ID == 20 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: FindSim finished." >> $ORIG_DIR/job_info.txt
fi
