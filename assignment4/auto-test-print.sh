#!/bin/bash

rm ./slurm-* ./*_blksz; sbatch -N 1 --ntasks-per-node=2 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 3s

rm ./*_blksz; sbatch -N 1 --ntasks-per-node=4 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 3s

rm ./*_blksz; sbatch -N 1 --ntasks-per-node=8 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 6s

rm ./*_blksz; sbatch -N 1 --ntasks-per-node=16 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 30s

rm ./*_blksz; sbatch -N 1 --ntasks-per-node=32 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 30s

rm ./*_blksz; sbatch -N 2 --ntasks-per-node=32 --gres=gpu:1 -t 30 ./slurmSpectrum.sh
sleep 30s
rm ./*_blksz

bash ./auto-print.sh