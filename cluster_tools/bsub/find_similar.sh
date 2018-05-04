#!/bin/bash -l

# Number of Array tasks: 20
# ALSO CHANGE BELOW IN TWO POSITIONS !!!

ORIG_DIR=/home/users/axel.pahl/cp

source activate chem
sleep 5

if [[ $LSB_JOBINDEX == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: FindSim started..."
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: FindSim started..." >> $ORIG_DIR/job_info.txt
  # remove tmp dir, start fresh
  if [ -d /scratch/apahl/cp/profiles/data/tmp ]; then
    rm -rf /scratch/apahl/cp/profiles/data/tmp
  fi
  if [ -d /scratch/apahl/cp/profiles/data/plots ]; then
    rm -rf /scratch/apahl/cp/profiles/data/plots
  fi
  if [ -d /scratch/apahl/cp/profiles/reports ]; then
    rm -rf /scratch/apahl/cp/profiles/reports
  fi
  rm -rf /scratch/apahl/cp/profiles/data/sim_refs-*.tsv
else
  sleep 180
fi

PLATES=$(get_plates)
find_similar -p $PLATES -t $LSB_JOBINDEX -n 20
source deactivate

if [[ $LSB_JOBINDEX == 20 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: FindSim finished." >> $ORIG_DIR/job_info.txt
fi
