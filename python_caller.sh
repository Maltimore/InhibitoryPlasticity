#!/bin/sh
#$ -V -cwd
#$ -q cognition-all.q
#$ -t 1-2
#$ -N mytest
#$ -e ./errors/$JOB_NAME.task_id_$TASK_ID
#$ -o ./text_output/$JOB_NAME.task_id_$TASK_ID
echo "I'm running task $SGE_TASK_ID on node $HOSTNAME!"
python simulation_cpp.py $SGE_TASK_ID
