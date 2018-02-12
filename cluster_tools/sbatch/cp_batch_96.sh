#!/bin/bash -l

#SBATCH --job-name=cp96
#SBATCH --array=1-96
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/cp96_%A-%a.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/cp96_%A-%a.txt
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=3000
# #SBATCH --time=24:00:00

CPPIPE=171213_mpc
FOLDER=${1%/}  # removes trailing slash, if there is one
INPUT=/ptmp/apahl/cp/queue/$FOLDER
OUTPUT=/ptmp/apahl/cp/output/${FOLDER}
ORIG_DIR=$(pwd)
JOB_LOG=$ORIG_DIR/logs/job_${SLURM_JOB_ID}_${FOLDER}.log
CELLPROF_DIR=/u/apahl/dev/github/CellProfiler

if [[ $SLURM_ARRAY_TASK_ID == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CP96 $FOLDER (pipe: $CPPIPE) started..."
  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CP96 $FOLDER (pipe: $CPPIPE) started..." >> $ORIG_DIR/job_info.txt
  mkdir -p $OUTPUT
fi

source activate cellprof
sleep 10
cellprofiler -c -p /ptmp/apahl/cp/conf/$CPPIPE.cppipe -r -i $INPUT -o $OUTPUT/$(((SLURM_ARRAY_TASK_ID - 1) * 36 + 1)) -L 10 -f $(((SLURM_ARRAY_TASK_ID - 1) * 36 + 1)) -l $((SLURM_ARRAY_TASK_ID * 36)) -t /ptmp/apahl/cp/tmp
RETVAL=$?
source deactivate

if [[ $RETVAL == 0 ]]; then
  RETSTAT="finished."
else
  RETSTAT="FAILED with error code $RETVAL."
fi

# add a line to the file which is used to keep track of the finished tasks:
echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID  $SLURM_ARRAY_TASK_ID  $RETSTAT" >> $JOB_LOG

if [[ $SLURM_ARRAY_TASK_ID == 96 ]]; then
  # copy the pipeline that was used into the output folder:
  cp /ptmp/apahl/cp/conf/$CPPIPE.cppipe $OUTPUT/
  # put the plate name also in the folder:
  echo $FOLDER > $OUTPUT/plate_info.txt
  # note the CellProfiler commit that was used
  cd $CELLPROF_DIR
  echo "CellProfiler Version Commit:" > $OUTPUT/versions.txt
  echo "$(git rev-parse --short HEAD)  ($(git show -s --format=%ci HEAD))" >> $OUTPUT/versions.txt
  cd $ORIG_DIR

  echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: CP96 $FOLDER finished." >> $ORIG_DIR/job_info.txt
fi
