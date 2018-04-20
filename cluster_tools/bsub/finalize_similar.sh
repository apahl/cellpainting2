#!/bin/bash -l

set -e
ORIG_DIR=/home/users/axel.pahl/cp

source activate chem
sleep 300
finalize_similar
source deactivate

echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: FinalSim done."
echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: FinalSim done." >> $ORIG_DIR/job_info.txt
