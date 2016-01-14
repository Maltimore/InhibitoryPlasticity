#!/bin/bash
#$ -V -cwd
#$ -q cognition-all.q
#$ -l h_vmem=5G
#$ -e ./errors/$JOB_NAME
#$ -o ./text_output/$JOB_NAME
echo "Running single simulation with name $JOB_NAME and lookupindex ${lookupindex}"
python simulation_cpp.py ${lookupindex} $JOB_NAME
