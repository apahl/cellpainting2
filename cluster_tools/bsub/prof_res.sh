#!/bin/bash -l

set -e
ORIG_DIR=/home/users/axel.pahl/cp

# remove existing data, start fresh
rm -rf /ptmp/apahl/cp/profiles/data
mkdir -p /ptmp/apahl/cp/profiles/data
rm -rf /ptmp/apahl/cp/profiles/reports

source activate chem
sleep 5
profile_results
source deactivate

echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: ProfRes done."
echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: ProfRes done." >> $ORIG_DIR/job_info.txt
