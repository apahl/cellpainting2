#!/bin/bash
OUTPUT=$HOME/sacct.txt
sacct > $OUTPUT
COMPL=$(printf "%3d" $(cat $OUTPUT | grep COMPL | wc -l))
RUNN=$(printf "%3d" $(cat $OUTPUT | grep RUNN | wc -l))
PEND=$(printf "%3d" $(cat $OUTPUT | grep PEND | wc -l))
FAIL=$(printf "%3d" $(cat $OUTPUT | grep FAIL | wc -l))
echo "Number of jobs completed: $COMPL"
echo "               running:   $RUNN"
echo "               pending:   $PEND"
echo "               failed:    $FAIL"
