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

echo "                     Job Overview"
echo "              Compl.  Runn.  Fail  Pend."
echo "cp_batch_96    $CP_COMPL     $CP_RUNN    $CP_FAIL   $CP_PEND"
echo "postproc       $PP_COMPL     $PP_RUNN    $PP_FAIL   $PP_PEND"