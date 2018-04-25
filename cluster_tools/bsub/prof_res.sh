#!/bin/bash -l

set -e
ORIG_DIR=/home/users/axel.pahl/cp

# remove existing data, start fresh
if [ -d /scratch/apahl/cp/profiles/data ]; then
  rm -rf /scratch/apahl/cp/profiles/data
fi
mkdir -p /scratch/apahl/cp/profiles/data
if [ -d /scratch/apahl/cp/profiles/reports ]; then
  rm -rf /scratch/apahl/cp/profiles/reports
fi

source activate chem
sleep 5
profile_results
source deactivate

echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: ProfRes done."
echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: ProfRes done." >> $ORIG_DIR/job_info.txt
