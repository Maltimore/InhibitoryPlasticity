#!/bin/bash

echo "Shall I clear the folders with output and error text? [y/n]"
read answer

if [ "$answer" == "y" ]; then
	echo "Clearing.."
	rm -rf ./text_output
	rm -rf ./errors
fi

if [ ! -d "./text_output" ]; then
	echo "Creating directory text_output"
	mkdir ./text_output
fi

if [ ! -d "./errors" ]; then
	echo "Creating directory errors"
	mkdir ./errors
fi

qsub python_caller.sh
