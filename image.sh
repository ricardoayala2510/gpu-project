#!/bin/bash
#SBATCH -J imageCNN               # Job name
#SBATCH -o image_run_%j.out       # Standard output file (%j = job ID)
#SBATCH -e image_run_%j.err       # Error output file
#SBATCH -p gpu-a100-dev           # Queue/partition (A100 GPU development)
#SBATCH -N 1                      # Number of nodes
#SBATCH -n 1                      # Number of tasks
#SBATCH -t 02:00:00               # Maximum runtime (2 hours)
#SBATCH --mail-type=END,FAIL      # Email notifications
#SBATCH --mail-user=ayalaacurio1025@my.msutexas.edu
#SBATCH -A ASC23018               # Project allocation ID

# -------------------------------------------
#  Load Required Modules
# -------------------------------------------
module purge
module load cuda/12.2             # or cuda/11.3 if required by the class

# -------------------------------------------
#  Environment Info
# -------------------------------------------
echo "-------------------------------------------"
echo "CUDA CNN Training Job Started"
echo "Date and Time: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA version:"
nvcc --version
echo "-------------------------------------------"

# -------------------------------------------
#  Check and Compile CUDA File
# -------------------------------------------
echo "Checking for CUDA source file..."
ls -l image.cu || { echo "image.cu not found!"; exit 1; }

echo "Compiling CUDA program..."
nvcc image.cu -o image.out -O2

if [ ! -f image.out ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"
ls -lh image.out

# -------------------------------------------
#  Run the Program on GPU
# -------------------------------------------
echo "-------------------------------------------"
echo "Starting GPU Execution..."
date
echo "-------------------------------------------"

# Save output both in the Slurm .out file and in a per-job log file
./image.out 2>&1 | tee output_log_${SLURM_JOB_ID}.txt

echo "-------------------------------------------"
echo "GPU Execution Finished"
date
echo "Output saved in: output_log_${SLURM_JOB_ID}.txt"
echo "-------------------------------------------"

echo "Job completed successfully!"
