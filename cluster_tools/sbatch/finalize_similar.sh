#!/bin/bash -l

#SBATCH --job-name=finalsim
#SBATCH --workdir=/ptmp/apahl/cp
#SBATCH --output=/ptmp/apahl/cp/jobout/finalsim_%j.txt
#SBATCH --error=/ptmp/apahl/cp/jobout/finalsim_%j.txt
#SBATCH --partition=express
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
# Memory usage of the job [MB]
#SBATCH --mem=4096
# #SBATCH --time=24:00:00

set -e
ORIG_DIR=$(pwd)

source activate chem
sleep 5
finalize_similar
source deactivate

echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: FinalSim done."
echo "`date +"%Y%m%d %H:%M"`  $SLURM_JOB_ID: FinalSim done." >> $ORIG_DIR/job_info.txt

# Dependency management:
# $ sbatch job1.sh
# 11254323

# $ sbatch --dependency=afterok:11254323 job2.sh