#!/bin/bash
COMPL=$(printf "%3d" $(sacct | grep COMPL | wc -l))
RUNN=$(printf "%3d" $(sacct | grep RUNN | wc -l))
PEND=$(printf "%3d" $(sacct | grep PEND | wc -l))
FAIL=$(printf "%3d" $(sacct | grep FAIL | wc -l))
echo "Number of jobs completed: $COMPL"
echo "               running:   $RUNN"
echo "               pending:   $PEND"
echo "               failed:    $FAIL"
