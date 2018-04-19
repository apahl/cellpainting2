#!/bin/bash -l

CPPIPE=180419_mpc
FOLDER=${1%/}  # removes trailing slash, if there is one
ORIG_DIR=/home/users/axel.pahl/cp
INPUT=$ORIG_DIR/queue/$FOLDER
OUTPUT=$ORIG_DIR/output/${FOLDER}
JOB_LOG=$ORIG_DIR/logs/job_${LSB_JOBID}_${FOLDER}.log
CELLPROF_DIR=/home/users/axel.pahl/dev/github/CellProfiler

if [[ $LSB_JOBINDEX == 1 ]]; then
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: CP96 $FOLDER (pipe: $CPPIPE) started..."
  echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: CP96 $FOLDER (pipe: $CPPIPE) started..." >> $ORIG_DIR/job_info.txt
  mkdir -p $OUTPUT
fi

source activate cellprof
sleep 10
cellprofiler -c -p $ORIG_DIR/conf/$CPPIPE.cppipe -r -i $INPUT -o $OUTPUT/$(((LSB_JOBINDEX - 1) * 36 + 1)) -L 10 -f $(((LSB_JOBINDEX - 1) * 36 + 1)) -l $((LSB_JOBINDEX * 36)) -t $ORIG_DIR/tmp
RETVAL=$?
source deactivate

if [[ $RETVAL == 0 ]]; then
  RETSTAT="finished."
else
  RETSTAT="FAILED with error code $RETVAL."
fi

# add a line to the file which is used to keep track of the finished tasks:
echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID  $LSB_JOBINDEX  $RETSTAT" >> $JOB_LOG

if [[ $LSB_JOBINDEX == 1 ]]; then
  # copy the pipeline that was used into the output folder:
  cp $ORIG_DIR/conf/$CPPIPE.cppipe $OUTPUT/
  # put the plate name also in the folder:
  echo $FOLDER > $OUTPUT/plate_info.txt
  # note the CellProfiler commit that was used
  cd $CELLPROF_DIR
  echo "CellProfiler Version Commit:" > $OUTPUT/versions.txt
  echo "$(git rev-parse --short HEAD)  ($(git show -s --format=%ci HEAD))" >> $OUTPUT/versions.txt
  cd $ORIG_DIR

fi
