#!/bin/bash
# This script runs a Python command 20 times

for i in $(seq 1 82)
do
    echo "Running iteration $i"
    python train_fgnet.py --split $i
done
