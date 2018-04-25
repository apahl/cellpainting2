#!/bin/bash -l

set -e
PLATE=${1%/}  # removes trailing slash, if there is one
PREFIX=/scratch/apahl/cp
OUTPUT=$PREFIX/output
FOLDER=$OUTPUT/${PLATE}
TARRESULTS=${PLATE}_results.tgz
TAROUTPUT=${PLATE}_output.tgz
ORIG_DIR=/home/users/axel.pahl/cp

if [[ $PLATE == "" ]]; then
  echo "Missing parameter PLATE."
  exit 1
fi

# SAMPLE IMAGES
echo "sampling images..."
sample_images $PLATE $PREFIX

# AGGREGATE the individual results and calc MEDIANS
echo ""
echo "aggregating results..."
source activate postproc
sleep 5
agg_results $FOLDER -t median -j 96
source deactivate

# TAR the RESULTS
echo ""
echo "taring results..."
cd $OUTPUT
if [ ! -e $PLATE/*.xml ]; then
  echo "No XML file found."
  echo "  creating dummy file."
  echo "dummy" > $PLATE/dummy.xml
fi
tar czf $TARRESULTS $PLATE/*.txt $PLATE/*.tsv $PLATE/*.cppipe $PLATE/*.xml $PLATE/images

# TAR OUTPUT
echo ""
echo "taring output..."
tar czf $TAROUTPUT $PLATE/
cd $ORIG_DIR

echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: PostProcessing $PLATE done."
echo "`date +"%Y%m%d %H:%M"`  $LSB_JOBID: PostProcessing $PLATE done." >> $ORIG_DIR/job_info.txt
