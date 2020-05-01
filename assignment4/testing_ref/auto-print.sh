#!/bin/bash

for i in slurm-*
do
    echo "--------------------------------------"
    echo "$i:"
    echo ""
    tail -n 15 "$i"
    echo ""
done