// ------------------------------------------------------------
// image.cu
// Simple CUDA-based CNN-style pipeline for PetImages (Cats vs Dogs)
// - Loads images from PetImages/Cat and PetImages/Dog
// - Normalizes pixels
// - Applies 3x3 conv filters + ReLU
// - Max-pools
// - Dense layer with sigmoid (binary classification)
// - Uses a very simple "Adam-like" update for the dense weights
// - Times each kernel and prints total program runtime
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

// stb_image: header-only image loader (JPG/PNG/etc.)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ------------------- Model / Training Hyperparameters -------------------
#define IMG_W 150         // assumed image width (pixels)
#define IMG_H 150         // assumed image height (pixels)
#define FILTERS 8         // number of convolution filters
#define KSIZE 3           // convolution kernel size (3x3)
#define TILE 16           // CUDA tile/block size
#define POOL 2            // max pooling factor (2x2)
#define LR 0.001f         // learning rate for Adam
#define BETA1 0.9f        // Adam beta1
#define BETA2 0.999f      // Adam beta2
#define EPS 1e-8f         // Adam epsilon
#define MAX_IMAGES 2800   // maximum images to load
#define TRAIN_RATIO 0.8f  // percentage of images used for training
#define EPOCHS 10         // number of training epochs

//  Global Timing Accumulators (Host)
// These accumulate total time spent (in ms) in each kernel across the run.
float tNormalize = 0.0f;
float tConv      = 0.0f;
float tPool      = 0.0f;
float tDense     = 0.0f;
float tLoss      = 0.0f;
float tAdam      = 0.0f;

// ========================================================================
//                              CUDA KERNELS
// ========================================================================

// Normalize unsigned char pixels [0..255] to floats [0..1]
__global__ void normalizeKernel(unsigned char *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / 255.0f;
}

// 3x3 convolution + ReLU, applied to a single-channel input
// x:   input image (w x h), single channel
// f:   FILTERS x 3x3 filter bank
// y:   output tensor with shape (FILTERS, h, w) flattened
__global__ void convKernel(float *x, float *f, float *y, int w, int h, int outc) {
    int X = blockIdx.x * TILE + threadIdx.x;
    int Y = blockIdx.y * TILE + threadIdx.y;
    if (X >= w || Y >= h) return;

    for (int oc = 0; oc < outc; oc++) {
        float s = 0.0f;
        // 3x3 kernel centered at (X, Y)
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int ix = min(max(X + kx, 0), w - 1); // clamp to image
                int iy = min(max(Y + ky, 0), h - 1);
                s += x[iy * w + ix] * f[oc * 9 + (ky + 1) * 3 + (kx + 1)];
            }
        }
        // ReLU activation
        y[oc * w * h + Y * w + X] = fmaxf(s, 0.0f);
    }
}

// 2x2 max pooling on each channel
// in:  FILTERS x (h x w)
// out: FILTERS x (h/POOL x w/POOL)
__global__ void maxPoolKernel(float *in, float *out, int w, int h, int c) {
    int X = blockIdx.x * TILE + threadIdx.x;
    int Y = blockIdx.y * TILE + threadIdx.y;
    if (X >= w / POOL || Y >= h / POOL) return;

    for (int ch = 0; ch < c; ch++) {
        float m = -1e9f;
        // 2x2 window
        for (int py = 0; py < POOL; py++) {
            for (int px = 0; px < POOL; px++) {
                int ix = X * POOL + px;
                int iy = Y * POOL + py;
                m = fmaxf(m, in[(ch * h + iy) * w + ix]);
            }
        }
        out[(ch * (h / POOL) + Y) * (w / POOL) + X] = m;
    }
}

// Dense layer (fully connected) with sigmoid activation.
// This version is intentionally simple and single-threaded.
__global__ void denseSigmoid(float *in, float *W, float *b, float *out, int len) {
    float s = *b;
    for (int i = 0; i < len; i++) s += in[i] * W[i];
    *out = 1.0f / (1.0f + expf(-s));
}

// Binary cross-entropy loss for a single prediction and label.
__global__ void binaryCrossEntropy(float *pred, int label, float *loss) {
    *loss = -label * logf(*pred + 1e-7f) - (1 - label) * logf(1 - *pred + 1e-7f);
}

// Adam optimizer update for dense weights
__global__ void adamUpdate(float *W, float *m, float *v, float *grad, int n, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];
        float mhat = m[i] / (1 - BETA1);
        float vhat = v[i] / (1 - BETA2);
        W[i] -= lr * mhat / (sqrtf(vhat) + EPS);
    }
}

// ========================================================================
//                           HELPER FUNCTIONS
// ========================================================================
//
// load_images:
//   - Reads all .jpg/.JPG/.png/.PNG files in 'dir'
//   - Builds full paths in 'paths'
//   - Assigns 'label_value' into labels[] for each loaded image
//   - Returns new count (starting from 'start')
//
int load_images(const char *dir,
                char paths[][512],
                int labels[],
                int start,
                int max,
                int label_value) {
    DIR *d;
    struct dirent *ent;
    int c = start;

    d = opendir(dir);
    if (!d) {
        // If directory does not exist or cannot be opened, print why
        perror(dir);
        return c;
    }

    while ((ent = readdir(d)) && c < max) {
        // Skip "." and ".."
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;

        // Accept JPG/PNG (case-insensitive)
        if (strstr(ent->d_name, ".jpg") || strstr(ent->d_name, ".JPG") ||
            strstr(ent->d_name, ".png") || strstr(ent->d_name, ".PNG")) {

            // Build full path: dir/filename
            snprintf(paths[c], 512, "%s/%s", dir, ent->d_name);
            // Assign fixed label for this directory (e.g., cat=0, dog=1)
            labels[c] = label_value;
            c++;
        }
    }
    closedir(d);
    return c;
}

// ========================================================================
//                               MAIN PROGRAM
// ========================================================================
int main() {
    printf("\n--------------------------------------------\n");
    printf(" CUDA CNN Training (PetImages Cat vs Dog)\n");
    printf("--------------------------------------------\n");

    // Total program timer (CPU wall clock) 
    clock_t progStart = clock();

    // Arrays to store file paths and labels for each image
    char paths[MAX_IMAGES][512];
    int  labels[MAX_IMAGES];

    //  Load Cat (label 0) and Dog (label 1) images
    int n = 0;
    n = load_images("PetImages/Cat", paths, labels, n, MAX_IMAGES, 0); // cats = 0
    n = load_images("PetImages/Dog", paths, labels, n, MAX_IMAGES, 1); // dogs = 1

    if (n == 0) {
        printf("❌ No images found.\n");
        return 0;
    }

    // Split into train and validation sets
    int trainN = (int)(n * TRAIN_RATIO);
    int valN   = n - trainN;
    printf("✅ Loaded %d images (Train %d, Validation %d)\n\n", n, trainN, valN);

    // Image and feature sizes
    int w = IMG_W, h = IMG_H;
    int size = w * h;                               // pixels per image
    int len  = (w / POOL) * (h / POOL) * FILTERS;   // length of flattened pooled feature map

    // Allocate and initialize convolution filters
    float *d_filter;
    cudaMalloc(&d_filter, FILTERS * 9 * sizeof(float));
    float *h_filter = (float*)malloc(FILTERS * 9 * sizeof(float));

    // Random small initialization for filters
    for (int i = 0; i < FILTERS * 9; i++)
        h_filter[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    cudaMemcpy(d_filter, h_filter, FILTERS * 9 * sizeof(float), cudaMemcpyHostToDevice);

    //Dense layer parameters and Adam buffers
    float *W, *b, *m, *v, *grad, *d_pred, *d_loss;
    cudaMalloc(&W,     len * sizeof(float));  // weights
    cudaMalloc(&b,     sizeof(float));        // bias
    cudaMalloc(&m,     len * sizeof(float));  // Adam first moment
    cudaMalloc(&v,     len * sizeof(float));  // Adam second moment
    cudaMalloc(&grad,  len * sizeof(float));  // gradient buffer
    cudaMalloc(&d_pred, sizeof(float));       // prediction
    cudaMalloc(&d_loss, sizeof(float));       // loss

    // Initialize dense weights randomly
    float *hW = (float*)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++)
        hW[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    cudaMemcpy(W, hW, len * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize bias to zero
    float hb = 0.0f;
    cudaMemcpy(b, &hb, sizeof(float), cudaMemcpyHostToDevice);

    // CUDA grid and block configuration for conv/pooling
    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);

    // Events reused to time each kernel invocation
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);

    printf("🚀 Training started...\n\n");

    // =====================================================================
    //                          TRAINING EPOCHS
    // =====================================================================
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        float totalLoss = 0.0f, totalAcc = 0.0f;
        float valLoss   = 0.0f, valAcc  = 0.0f;

        // Time the entire training loop of this epoch
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Training loop
        for (int idx = 0; idx < trainN; idx++) {
            int label = labels[idx];  // 0 for cat, 1 for dog

            int ww, hh, ch;
            // Load image as grayscale (1 channel)
            unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);
            if (!img) continue;  // if loading failed, skip

            // Device buffers for this image
            unsigned char *d_img;
            float *d_norm, *d_conv, *d_pool;
            cudaMalloc(&d_img,  size);
            cudaMalloc(&d_norm, size * sizeof(float));
            cudaMalloc(&d_conv, FILTERS * size * sizeof(float));
            cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));

            // Copy raw pixels to device (note: assumes ww*hh == size or compatible)
            cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

            float msKernel;

            // normalizeKernel 
            cudaEventRecord(kStart);
            normalizeKernel<<<(size + 255) / 256, 256>>>(d_img, d_norm, size);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tNormalize += msKernel;

            // convKernel 
            cudaEventRecord(kStart);
            convKernel<<<grid, block>>>(d_norm, d_filter, d_conv, w, h, FILTERS);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tConv += msKernel;

            //  maxPoolKernel 
            cudaEventRecord(kStart);
            maxPoolKernel<<<grid, block>>>(d_conv, d_pool, w, h, FILTERS);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tPool += msKernel;

            // denseSigmoid 
            cudaEventRecord(kStart);
            denseSigmoid<<<1, 1>>>(d_pool, W, b, d_pred, len);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tDense += msKernel;

            //  binaryCrossEntropy 
            cudaEventRecord(kStart);
            binaryCrossEntropy<<<1, 1>>>(d_pred, label, d_loss);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tLoss += msKernel;

            // Fetch loss and prediction back to host
            float loss, pred;
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);

            totalLoss += loss;
            if ((pred > 0.5f) == label) totalAcc += 1.0f;

            // gradient & Adam update 
            // Very simplified gradient: every weight sees the same (pred - label).
            float g = (pred - label);
            float *hgrad = (float*)malloc(len * sizeof(float));
            for (int i = 0; i < len; i++) hgrad[i] = g;
            cudaMemcpy(grad, hgrad, len * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(kStart);
            adamUpdate<<<(len + 255) / 256, 256>>>(W, m, v, grad, len, LR);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tAdam += msKernel;

            free(hgrad);

            // Free per-image device buffers
            cudaFree(d_img);
            cudaFree(d_norm);
            cudaFree(d_conv);
            cudaFree(d_pool);
            stbi_image_free(img);
        }

        // End of training loop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Validation loop
        for (int idx = trainN; idx < n; idx++) {
            int label = labels[idx];

            int ww, hh, ch;
            unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);
            if (!img) continue;

            unsigned char *d_img;
            float *d_norm, *d_conv, *d_pool;
            cudaMalloc(&d_img,  size);
            cudaMalloc(&d_norm, size * sizeof(float));
            cudaMalloc(&d_conv, FILTERS * size * sizeof(float));
            cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));
            cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

            float msKernel;

            // normalizeKernel (validation)
            cudaEventRecord(kStart);
            normalizeKernel<<<(size + 255) / 256, 256>>>(d_img, d_norm, size);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tNormalize += msKernel;

            // convKernel (validation)
            cudaEventRecord(kStart);
            convKernel<<<grid, block>>>(d_norm, d_filter, d_conv, w, h, FILTERS);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tConv += msKernel;

            // maxPoolKernel (validation)
            cudaEventRecord(kStart);
            maxPoolKernel<<<grid, block>>>(d_conv, d_pool, w, h, FILTERS);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tPool += msKernel;

            // denseSigmoid (validation)
            cudaEventRecord(kStart);
            denseSigmoid<<<1, 1>>>(d_pool, W, b, d_pred, len);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tDense += msKernel;

            // binaryCrossEntropy (validation)
            cudaEventRecord(kStart);
            binaryCrossEntropy<<<1, 1>>>(d_pred, label, d_loss);
            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            cudaEventElapsedTime(&msKernel, kStart, kStop);
            tLoss += msKernel;

            float loss, pred;
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);
            valLoss += loss;
            if ((pred > 0.5f) == label) valAcc += 1.0f;

            cudaFree(d_img);
            cudaFree(d_norm);
            cudaFree(d_conv);
            cudaFree(d_pool);
            stbi_image_free(img);
        }

        // Epoch summary
        printf("Epoch %d/%d — %.1fs %.3fms/step — accuracy: %.4f — loss: %.4f — "
               "val_accuracy: %.4f — val_loss: %.4f\n",
               epoch, EPOCHS, ms / 1000.0f, ms / trainN,
               totalAcc / trainN, totalLoss / trainN, valAcc / valN, valLoss / valN);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n✅ Training complete!\n");

    // Kernel timing summary 
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);

    printf("\n==== Kernel time summary (GPU) ====\n");
    printf("normalizeKernel: %.3f ms\n", tNormalize);
    printf("convKernel     : %.3f ms\n", tConv);
    printf("maxPoolKernel  : %.3f ms\n", tPool);
    printf("denseSigmoid   : %.3f ms\n", tDense);
    printf("binaryCrossEnt.: %.3f ms\n", tLoss);
    printf("adamUpdate     : %.3f ms\n", tAdam);

    // Total program time 
    clock_t progEnd = clock();
    double progSecs = (double)(progEnd - progStart) / CLOCKS_PER_SEC;
    printf("\nTotal program time (CPU wall clock): %.3f s\n", progSecs);

    // Cleanup global allocations
    cudaFree(W); cudaFree(b); cudaFree(m); cudaFree(v);
    cudaFree(grad); cudaFree(d_filter); cudaFree(d_pred); cudaFree(d_loss);
    free(h_filter); free(hW);

    return 0;
}
