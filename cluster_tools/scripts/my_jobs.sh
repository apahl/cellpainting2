#!/bin/bash
OUTPUT=$HOME/sacct.txt
sacct > $OUTPUT
CP_COMPL=$(printf "%3d" $(cat $OUTPUT | grep cp96 | grep COMPL | wc -l))
CP_RUNN=$(printf "%3d" $(cat $OUTPUT | grep cp96 | grep RUNN | wc -l))
CP_PEND=$(printf "%3d" $(cat $OUTPUT | grep cp96 | grep PEND | wc -l))
CP_FAIL=$(printf "%3d" $(cat $OUTPUT | grep cp96 | grep FAIL | wc -l))

PP_COMPL=$(printf "%3d" $(cat $OUTPUT | grep postproc | grep COMPL | wc -l))
PP_RUNN=$(printf "%3d" $(cat $OUTPUT | grep postproc | grep RUNN | wc -l))
PP_PEND=$(printf "%3d" $(cat $OUTPUT | grep postproc | grep PEND | wc -l))
PP_FAIL=$(printf "%3d" $(cat $OUTPUT | grep postproc | grep FAIL | wc -l))

PR_COMPL=$(printf "%3d" $(cat $OUTPUT | grep profres | grep COMPL | wc -l))
PR_RUNN=$(printf "%3d" $(cat $OUTPUT | grep profres | grep RUNN | wc -l))
PR_PEND=$(printf "%3d" $(cat $OUTPUT | grep profres | grep PEND | wc -l))
PR_FAIL=$(printf "%3d" $(cat $OUTPUT | grep profres | grep FAIL | wc -l))

FS_COMPL=$(printf "%3d" $(cat $OUTPUT | grep findsim | grep COMPL | wc -l))
FS_RUNN=$(printf "%3d" $(cat $OUTPUT | grep findsim | grep RUNN | wc -l))
FS_PEND=$(printf "%3d" $(cat $OUTPUT | grep findsim | grep PEND | wc -l))
FS_FAIL=$(printf "%3d" $(cat $OUTPUT | grep findsim | grep FAIL | wc -l))

CR_COMPL=$(printf "%3d" $(cat $OUTPUT | grep create | grep COMPL | wc -l))
CR_RUNN=$(printf "%3d" $(cat $OUTPUT | grep create | grep RUNN | wc -l))
CR_PEND=$(printf "%3d" $(cat $OUTPUT | grep create | grep PEND | wc -l))
CR_FAIL=$(printf "%3d" $(cat $OUTPUT | grep create | grep FAIL | wc -l))

echo "                     Job Overview"
echo "              Compl.  Runn.  Fail  Pend."
if [ $CP_COMPL -ne 0 ] || [ $CP_RUNN -ne 0 ] || [ $CP_FAIL -ne 0 ] || [ $CP_PEND -ne 0 ]; then
  echo "cp_batch_96    $CP_COMPL     $CP_RUNN    $CP_FAIL   $CP_PEND"
fi
if [ $PP_COMPL -ne 0 ] || [ $PP_RUNN -ne 0 ] || [ $PP_FAIL -ne 0 ] || [ $PP_PEND -ne 0 ]; then
  echo "postproc       $PP_COMPL     $PP_RUNN    $PP_FAIL   $PP_PEND"
fi
if [ $PR_COMPL -ne 0 ] || [ $PR_RUNN -ne 0 ] || [ $PR_FAIL -ne 0 ] || [ $PR_PEND -ne 0 ]; then
  echo "profres        $PR_COMPL     $PR_RUNN    $PR_FAIL   $PR_PEND"
fi
if [ $FS_COMPL -ne 0 ] || [ $FS_RUNN -ne 0 ] || [ $FS_FAIL -ne 0 ] || [ $FS_PEND -ne 0 ]; then
  echo "findsim        $FS_COMPL     $FS_RUNN    $FS_FAIL   $FS_PEND"
fi
if [ $CR_COMPL -ne 0 ] || [ $CR_RUNN -ne 0 ] || [ $CR_FAIL -ne 0 ] || [ $CR_PEND -ne 0 ]; then
  echo "createrp       $CR_COMPL     $CR_RUNN    $CR_FAIL   $CR_PEND"
fi
