# CUDA-Accelerated Cats vs Dogs CNN (TACC Lonestar6)

This repository contains a **CUDA/C++ implementation** of an image-classification pipeline for **binary classification (cats vs dogs)**, developed as a **two-person group project** and benchmarked on **TACC Lonestar 6 (LS6)** using **Slurm**.

The goal is to demonstrate how **GPU parallelism** can accelerate the core compute stages of a CNN-style workflow compared to a sequential CPU run.

---

## Project highlights

- **Runs on LS6 A100 GPUs** (Slurm job script included: `image.sh`)
- Implements key CNN-like stages in CUDA (e.g., normalization, convolution, pooling, dense, loss, Adam)
- Produces **timing summaries** for GPU kernels and end-to-end program time
- Includes a **CPU vs GPU performance comparison**
- Based on an original **Python ML project** (see reference below)

---

## Presentation

Project presentation (slides/demo):
- https://www.canva.com/design/DAG6OOoFiro/2YrGJky5tvQqMv2-ns_hmA/edit?utm_content=DAG6OOoFiro&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton&fbclid=PAVERFWAOyTJRleHRuA2FlbQIxMABzcnRjBmFwcF9pZA8xMjQwMjQ1NzQyODc0MTQAAach_-ECJKuP_qcaLy8CmUwLNnzAKFrRoReVTiSBQcXy9PDd7NfOCF9CvAeS-g_aem_3lDc-APUDKLz1lSOJnwJOQ

---

## Team (2 people)


- Member 1: `< Ricardo Ayala>`
- Member 2: `<Shishir Adikari>`

---

## Reference (original Python ML project)

This CUDA project is based on and inspired by the original Python project:

- https://github.com/Shishirise/ML-with-Python/tree/main/Project%201?fbclid=PAVERFWAOySgVleHRuA2FlbQIxMABzcnRjBmFwcF9pZA8xMjQwMjQ1NzQyODc0MTQAAafHwv3OwoEl2GF6UmXQtQIRx1bX0KmG7hguooZdBBzMNSZJWNirXQFW-bh2-Q_aem_uqBQDbS4ytJpEKV5qyFWqQ

---

## How to run on TACC Lonestar 6

### 1) Load modules

On LS6:
```bash
module purge
module load cuda/12.2
```

### 2) Submit the Slurm job

`image.sh` compiles and runs the program on an A100 GPU partition:
```bash
sbatch image.sh
```

The job script (included) does:
- `nvcc image.cu -o image.out -O2`
- `./image.out > output_${SLURM_JOB_ID}.txt`

---

## Performance comparison (CPU vs GPU)

Measured end-to-end (wall-clock) times:

- **CPU (sequential):** ~20 minutes ≈ **1200 s**
- **GPU (LS6 A100):** ~**220 s** (observed: **218.190 s**)

### Speedup

- **Speedup:** 1200 / 220 ≈ **5.45× faster**
- **Time saved:** 1200 − 220 = **980 s** (≈ **16 min 20 s**)
- **Runtime reduction:** 980 / 1200 ≈ **81.7% less time**

### Kernel-level insight (GPU)

The program reports a **total GPU kernel time** of **~15.494 s**, while the **total program wall time** is **~218.190 s**.

This gap suggests much of the end-to-end time is spent outside GPU kernels, such as:
- image loading / decoding
- host-side preprocessing
- CPU↔GPU memory transfers and synchronization
- any non-kernel bookkeeping

Optimizing those parts (batching transfers, overlap, pinned memory, faster I/O) can further reduce total runtime.

---

## Why this project is useful

- Demonstrates a practical **GPU acceleration workflow** on an HPC system (LS6 + Slurm)
- Shows how to instrument and interpret **kernel timings vs total runtime**
- Provides a clear real-world speedup compared to a sequential CPU implementation
- Useful as a learning reference for:
  - CUDA kernels and memory management
  - basic CNN pipeline stages
  - HPC job submission and benchmarking

---

## Files

- `image.cu` — main CUDA/C++ program
- `stb_image.h` — image loading header (STB)
- `image.sh` — Slurm script for LS6 execution
- `RESULTS.md` — detailed run outputs + findings

