#!/bin/bash

if [ ! -d "./text_output" ]; then
	echo "Creating directory text_output"
	mkdir ./text_output
fi

if [ ! -d "./errors" ]; then
	echo "Creating directory errors"
	mkdir ./errors
fi

echo "Calling python_caller_single.sh with lookupindex $2"
qsub -N "$1.task_id_$2" -v lookupindex=$2 python_caller_single.sh 
