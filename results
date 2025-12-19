# Findings and Results

This document summarizes the dataset stats, training logs, and **CPU vs GPU performance** based on the outputs you provided.

---

## Dataset summary

**Loaded:** 24,939 images  
- **Train:** 19,951 images  
- **Test:** 4,988 images  

The dataset contains **2 classes**: **cats** and **dogs**.

> If you see warnings like “Corrupt JPEG data … extraneous bytes”, they are usually **non-critical**.  
> They indicate some files have unusual metadata/extra bytes but are still readable in many pipelines (especially when allowing truncated reads).

---

## GPU run (CUDA on TACC Lonestar 6)

### Training output (GPU)

```
Loaded 24939 images (train 19951, test 4988)
Epoch 1/10 — 23s — 1.1ms/step — train_accuracy: 0.9971 — train_loss: 0.0395
Epoch 2/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0471
Epoch 3/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0472
Epoch 4/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0474
Epoch 5/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0472
Epoch 6/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0474
Epoch 7/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0472
Epoch 8/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0474
Epoch 9/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0472
Epoch 10/10 — 21s — 1.1ms/step — train_accuracy: 0.9964 — train_loss: 0.0474

Test Accuracy: 1.0000 — Test Loss: 0.0002
```

### GPU timing summary (kernel-level vs end-to-end)

```
==== FINAL GPU Kernel Time Summary ====
normalize: 1863.270 ms
conv     : 2754.268 ms
pool     : 2676.000 ms
dense    : 4524.844 ms
loss     : 1602.292 ms
adam     : 2073.112 ms
Total GPU kernel time: 15493.785 ms (15.494 s)
Total program time (CPU wall clock): 218.190 s
```

**Observation:** GPU kernels account for ~15.5 s, while total wall time is ~218.2 s.  
This indicates significant overhead outside kernels (I/O, CPU preprocessing, memory transfers, synchronization).

---

## CPU run (Sequential / CPU)

### Training output (CPU)

(Sequential CPU run):

```
Training started...
Epoch 1/10
624/624 — 108s 167ms/step — accuracy: 0.6005 — loss: 0.6709 — val_accuracy: 0.7379 — val_loss: 0.5212
Epoch 2/10
624/624 — 106s 170ms/step — accuracy: 0.7724 — loss: 0.4835 — val_accuracy: 0.8079 — val_loss: 0.4319
Epoch 3/10
624/624 — 111s 177ms/step — accuracy: 0.8275 — loss: 0.3879 — val_accuracy: 0.8233 — val_loss: 0.3889
Epoch 4/10
624/624 — 115s 184ms/step — accuracy: 0.8606 — loss: 0.3186 — val_accuracy: 0.8486 — val_loss: 0.3554
Epoch 5/10
624/624 — 111s 178ms/step — accuracy: 0.8980 — loss: 0.2420 — val_accuracy: 0.8442 — val_loss: 0.3730
Epoch 6/10
624/624 — 109s 175ms/step — accuracy: 0.9301 — loss: 0.1729 — val_accuracy: 0.8518 — val_loss: 0.4058
Epoch 7/10
624/624 — 110s 176ms/step — accuracy: 0.9499 — loss: 0.1271 — val_accuracy: 0.8516 — val_loss: 0.4741
Epoch 8/10
624/624 — 125s 201ms/step — accuracy: 0.9630 — loss: 0.0893 — val_accuracy: 0.8348 — val_loss: 0.6103
Epoch 9/10
624/624 — 131s 210ms/step — accuracy: 0.9740 — loss: 0.0707 — val_accuracy: 0.8392 — val_loss: 0.6256
Epoch 10/10
624/624 — 125s 200ms/step — accuracy: 0.9756 — loss: 0.0653 — val_accuracy: 0.8524 — val_loss: 0.6576
```

### What the CPU log shows

- **Training accuracy** steadily increases from **0.6005 → 0.9756**
- **Validation accuracy** peaks around **~0.8524** at the end
- **Validation loss** increases in later epochs (e.g., **0.6576** at epoch 10), suggesting the model may start to **overfit** as training continues.

---

## CPU vs GPU performance

End-to-end benchmark you provided:

- **CPU total time:** ~20 minutes ≈ **1200 s**
- **GPU total time:** ~**220 s** (observed: **218.190 s**)

### Speedup

- **Speedup:** 1200 / 220 ≈ **5.45×**
- **Time saved:** 980 s (≈ 16 min 20 s)
- **Runtime reduction:** ~81.7%

---

## Summary of key findings

- The GPU version significantly reduces overall runtime (**~5.45× faster**) compared to the CPU run.
- GPU kernels themselves execute quickly (**~15.5 s total**), and most remaining time is overhead (I/O + host work + transfers).
- CPU training reaches strong training accuracy (~0.976) but validation accuracy plateaus around ~0.85 and validation loss rises late in training.
- GPU run reports very high test performance (Test Accuracy: 1.0000, Test Loss: 0.0002) in the provided output.

