#!/bin/bash -l

#SBATCH --job-name=postproc
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/postproc_%j.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/postproc_%j.txt
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=2000
# #SBATCH --time=24:00:00

set -e
PLATE=${1%/}  # removes trailing slash, if there is one
PREFIX=/ptmp/apahl/cp
OUTPUT=$PREFIX/output
FOLDER=$OUTPUT/${PLATE}
TARRESULTS=${PLATE}_results.tgz
TAROUTPUT=${PLATE}_output.tgz
ORIG_DIR=$(pwd)

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
tar czf $TARRESULTS $PLATE/*.txt $PLATE/*.tsv $PLATE/*.cppipe $PLATE/images

# TAR OUTPUT
echo ""
echo "taring output..."
tar czf $TAROUTPUT $PLATE/
cd $ORIG_DIR

echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: PostProcessing $PLATE done."
echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: PostProcessing $PLATE done." >> $ORIG_DIR/job_info.txt
