#!/bin/bash

rm ./slurm-* ./*_blksz

for i in 2 4 8 16 32
do
    sbatch -N 1 --ntasks-per-node="$i" --gres=gpu:1 -t 30 ./slurmSpectrum.sh
done

sbatch -N 2 --ntasks-per-node=32 --gres=gpu:1 -t 30 ./slurmSpectrum.sh