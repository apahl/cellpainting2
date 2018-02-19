#!/bin/bash -l

#SBATCH --job-name=profres
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/profres_%j.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/profres_%j.txt
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=6144
# #SBATCH --time=24:00:00

set -e
ORIG_DIR=$(pwd)

# remove existing data, start fresh
rm -rf /ptmp/apahl/cp/profiles/data
mkdir -p /ptmp/apahl/cp/profiles/data
rm -rf /ptmp/apahl/cp/profiles/reports

source activate chem
sleep 5
profile_results
source deactivate

echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: ProfRes done."
echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: ProfRes done." >> $ORIG_DIR/job_info.txt
