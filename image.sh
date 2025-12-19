#!/bin/bash
#SBATCH -J image
#SBATCH -o image_%j.out
#SBATCH -e image_%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sadhikari0902@my.msutexas.edu
#SBATCH -A ASC23018

module purge
module load cuda/12.2

echo "Job started"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
nvcc --version

# Compile
nvcc image.cu -o image.out -O2

# Run
./image.out > output_${SLURM_JOB_ID}.txt

echo "Job finished"
date
