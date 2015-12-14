#!/bin/sh
#$ -V -cwd
#$ -q cognition-all.q
#$ -l h_vmem=5G
#$ -N maltes_sim
#$ -e ./errors/$JOB_NAME.task_id_16
#$ -o ./text_output/$JOB_NAME.task_id_16
python simulation_cpp.py 16 
