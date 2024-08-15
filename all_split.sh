#!/bin/bash
# This script runs a Python command 20 times

for i in $(seq 0 4)
do
    echo "Training on split $i"
    python train.py --split $i
done
