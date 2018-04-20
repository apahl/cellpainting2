#!/bin/bash -l

# Number of Array tasks: 20
# ALSO CHANGE BELOW IN TWO POSITIONS !!!

ORIG_DIR=/home/users/axel.pahl/cp

source activate chem
sleep 5

if [[ $LSB_JOBINDEX == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: CreateRp started..."
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: CreateRp started..." >> $ORIG_DIR/job_info.txt
  # remove existing report dir, start fresh
  rm -rf /ptmp/apahl/cp/profiles/reports
else
  sleep 120
fi

PLATES=$(get_plates)
create_reports -p $PLATES -t $LSB_JOBINDEX -n 20
source deactivate

if [[ $LSB_JOBINDEX == 20 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: CreateRp finished." >> $ORIG_DIR/job_info.txt
fi
