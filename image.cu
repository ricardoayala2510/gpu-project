// ------------------------------------------------------------
// image.cu
// CUDA CNN (Cats vs Dogs)
// - Parallel normalize / conv / pool / dense / loss / adam
// - 1024-thread parallel Dense reduction
// - Training + Testing output in Keras style
// - ADDED: Kernel time summary after each epoch
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ------------------ Hyperparameters ------------------
#define IMG_W 150
#define IMG_H 150
#define FILTERS 8
#define TILE 16
#define POOL 2
#define LR 0.001f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS 1e-8f
#define MAX_IMAGES 25000
#define TRAIN_RATIO 0.8f
#define EPOCHS 10

// GPU timing accumulators
float tNormalize = 0, tConv = 0, tPool = 0, tDense = 0, tLoss = 0, tAdam = 0;

// ======================================================
//                    KERNELS
// ======================================================

// Normalize
__global__ void normalizeKernel(unsigned char *in, float *out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) out[i] = in[i] * (1.0f/255.0f);
}

// Convolution
__global__ void convKernel(float *x, float *f, float *y, int w, int h, int oc){
    int X = blockIdx.x*TILE + threadIdx.x;
    int Y = blockIdx.y*TILE + threadIdx.y;
    if(X>=w || Y>=h) return;

    for(int k=0;k<oc;k++){
        float s = 0;
        for(int ky=-1; ky<=1; ky++)
        for(int kx=-1; kx<=1; kx++){
            int ix = min(max(X+kx,0), w-1);
            int iy = min(max(Y+ky,0), h-1);
            s += x[iy*w+ix] * f[k*9 + (ky+1)*3 + (kx+1)];
        }
        y[k*w*h + Y*w + X] = fmaxf(s,0.0f);
    }
}

// MaxPool 2×2
__global__ void maxPoolKernel(float *in, float *out, int w, int h, int c){
    int X = blockIdx.x*TILE + threadIdx.x;
    int Y = blockIdx.y*TILE + threadIdx.y;
    int W2 = w/2, H2 = h/2;

    if(X>=W2 || Y>=H2) return;

    for(int ch=0; ch<c; ch++){
        float m = -1e9f;
        for(int py=0; py<2; py++)
        for(int px=0; px<2; px++){
            int ix = X*2 + px;
            int iy = Y*2 + py;
            m = fmaxf(m, in[(ch*h + iy)*w + ix]);
        }
        out[(ch*H2 + Y)*W2 + X] = m;
    }
}

// ------------------ Dense 1024 threads ------------------

__global__ void dense_partial(float *input, float *W, float *partial, int n){
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + tid;

    float v = (i<n ? input[i]*W[i] : 0);
    s[tid] = v;
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride>>=1){
        if(tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }

    if(tid==0) partial[blockIdx.x] = s[0];
}

__global__ void dense_final(float *partial, float *bias, float *out, int n){
    extern __shared__ float s[];
    int tid = threadIdx.x;

    s[tid] = (tid<n ? partial[tid] : 0);
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride>>=1){
        if(tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }

    if(tid==0){
        float z = s[0] + *bias;
        out[0] = 1.0f / (1.0f + expf(-z));
    }
}

// Loss
__global__ void lossParallel(float *pred, int label, float *loss){
    float p = pred[0];
    loss[0] = -label*logf(p+1e-7f) - (1-label)*logf(1-p+1e-7f);
}

// Gradient (parallel)
__global__ void gradParallel(float *grad, float p, int label, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) grad[i] = (p - label);
}

// Adam
__global__ void adamUpdate(float *W,float *m,float *v,float *grad,int n,float lr){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
        m[i] = BETA1*m[i] + (1-BETA1)*grad[i];
        v[i] = BETA2*v[i] + (1-BETA2)*grad[i]*grad[i];
        float mh = m[i] / (1-BETA1);
        float vh = v[i] / (1-BETA2);
        W[i] -= lr * mh / (sqrtf(vh)+EPS);
    }
}


// ======================================================
// LOAD IMAGES
// ======================================================
int load_images(const char *dir, char paths[][512], int labels[],
                int start, int max, int val)
{
    DIR *d = opendir(dir);
    if(!d) return start;

    struct dirent *ent;
    int c=start;
    while((ent=readdir(d)) && c<max){
        if(strstr(ent->d_name,".jpg") || strstr(ent->d_name,".png")){
            snprintf(paths[c],512,"%s/%s",dir,ent->d_name);
            labels[c] = val;
            c++;
        }
    }
    closedir(d);
    return c;
}


// ======================================================
// MAIN
// ======================================================
int main(){

    clock_t progStart = clock();

    char paths[MAX_IMAGES][512];
    int labels[MAX_IMAGES];
    int n=0;

    n = load_images("PetImages/Cat", paths, labels, n, MAX_IMAGES, 0);
    n = load_images("PetImages/Dog", paths, labels, n, MAX_IMAGES, 1);

    if(n==0){ printf("No images found.\n"); return 0; }

    int trainN = n*TRAIN_RATIO;
    int testN  = n-trainN;

    printf("CUDA CNN Parallel 1024\n");
    printf("Loaded %d images (train %d, test %d)\n", n, trainN, testN);

    int size = IMG_W*IMG_H;
    int len  = (IMG_W/2)*(IMG_H/2)*FILTERS;

    float *W,*b,*m,*v,*grad,*d_pred,*d_loss;
    cudaMalloc(&W,len*sizeof(float));
    cudaMalloc(&b,sizeof(float));
    cudaMalloc(&m,len*sizeof(float));
    cudaMalloc(&v,len*sizeof(float));
    cudaMalloc(&grad,len*sizeof(float));
    cudaMalloc(&d_pred,sizeof(float));
    cudaMalloc(&d_loss,sizeof(float));

    float *hW=(float*)malloc(len*sizeof(float));
    for(int i=0;i<len;i++) hW[i]=((float)rand()/RAND_MAX-0.5f)*0.1f;
    float hb=0;

    cudaMemcpy(W,hW,len*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b,&hb,sizeof(float),cudaMemcpyHostToDevice);

    float *hf=(float*)malloc(FILTERS*9*sizeof(float));
    for(int i=0;i<FILTERS*9;i++) hf[i]=((float)rand()/RAND_MAX-0.5f)*0.1f;

    float *d_filter;
    cudaMalloc(&d_filter,FILTERS*9*sizeof(float));
    cudaMemcpy(d_filter,hf,FILTERS*9*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(TILE,TILE);
    dim3 grid((IMG_W+TILE-1)/TILE,(IMG_H+TILE-1)/TILE);

    cudaEvent_t ks,ke;
    cudaEventCreate(&ks);
    cudaEventCreate(&ke);

    // ---------------------- TRAINING ----------------------
    for(int e=1; e<=EPOCHS; e++){

        float TL=0, TA=0;
        clock_t epochStart = clock();

        for(int idx=0; idx<trainN; idx++){

            int label = labels[idx];
            int ww,hh,ch;
            unsigned char *img = stbi_load(paths[idx],&ww,&hh,&ch,1);
            if(!img) continue;

            unsigned char *d_img;
            float *d_norm,*d_conv,*d_pool;

            cudaMalloc(&d_img,size);
            cudaMalloc(&d_norm,size*sizeof(float));
            cudaMalloc(&d_conv,FILTERS*size*sizeof(float));
            cudaMalloc(&d_pool,FILTERS*(size/4)*sizeof(float));
            cudaMemcpy(d_img,img,size,cudaMemcpyHostToDevice);

            float ms;

            // Normalize
            cudaEventRecord(ks);
            normalizeKernel<<<(size+1023)/1024,1024>>>(d_img,d_norm,size);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tNormalize += ms;

            // Convolution
            cudaEventRecord(ks);
            convKernel<<<grid,block>>>(d_norm,d_filter,d_conv,IMG_W,IMG_H,FILTERS);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tConv += ms;

            // Pool
            cudaEventRecord(ks);
            maxPoolKernel<<<grid,block>>>(d_conv,d_pool,IMG_W,IMG_H,FILTERS);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tPool += ms;

            // Dense
            int threads=1024;
            int blocks=(len+threads-1)/threads;
            float *d_partial;
            cudaMalloc(&d_partial,blocks*sizeof(float));

            cudaEventRecord(ks);
            dense_partial<<<blocks,threads,threads*sizeof(float)>>>
                (d_pool,W,d_partial,len);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tDense += ms;

            cudaEventRecord(ks);
            dense_final<<<1,threads,threads*sizeof(float)>>>
                (d_partial,b,d_pred,blocks);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tDense += ms;

            cudaFree(d_partial);

            // Loss
            cudaEventRecord(ks);
            lossParallel<<<1,1>>>(d_pred,label,d_loss);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tLoss += ms;

            float loss,pred;
            cudaMemcpy(&loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(&pred,d_pred,sizeof(float),cudaMemcpyDeviceToHost);

            TL += loss;
            if((pred>0.5)==label) TA++;

            // Gradient + Adam
            gradParallel<<<blocks,threads>>>(grad,pred,label,len);

            cudaEventRecord(ks);
            adamUpdate<<<blocks,threads>>>(W,m,v,grad,len,LR);
            cudaEventRecord(ke);
            cudaEventSynchronize(ke);
            cudaEventElapsedTime(&ms,ks,ke);
            tAdam += ms;

            cudaFree(d_img);
            cudaFree(d_norm);
            cudaFree(d_conv);
            cudaFree(d_pool);
            stbi_image_free(img);
        }

        clock_t epochEnd = clock();
        float epochSec = (float)(epochEnd-epochStart)/CLOCKS_PER_SEC;
        float msPerStep = (epochSec*1000.0f)/trainN;

        // --------- EPOCH SUMMARY ---------
        printf("Epoch %d/%d — %.0fs — %.1fms/step — train_accuracy: %.4f — train_loss: %.4f\n",
               e,EPOCHS,epochSec,msPerStep,
               TA/trainN, TL/trainN);
    }

    // ---------------------- TESTING ----------------------
    float TL2=0, TA2=0;

    for(int idx=trainN; idx<n; idx++){
        int label = labels[idx];
        int ww,hh,ch;
        unsigned char *img = stbi_load(paths[idx],&ww,&hh,&ch,1);
        if(!img) continue;

        unsigned char *d_img;
        float *d_norm,*d_conv,*d_pool;
        cudaMalloc(&d_img,size);
        cudaMalloc(&d_norm,size*sizeof(float));
        cudaMalloc(&d_conv,FILTERS*size*sizeof(float));
        cudaMalloc(&d_pool,FILTERS*(size/4)*sizeof(float));

        cudaMemcpy(d_img,img,size,cudaMemcpyHostToDevice);

        normalizeKernel<<<(size+1023)/1024,1024>>>(d_img,d_norm,size);
        convKernel<<<grid,block>>>(d_norm,d_filter,d_conv,IMG_W,IMG_H,FILTERS);
        maxPoolKernel<<<grid,block>>>(d_conv,d_pool,IMG_W,IMG_H,FILTERS);

        int threads=1024;
        int blocks=(len+threads-1)/threads;
        float *d_partial;
        cudaMalloc(&d_partial,blocks*sizeof(float));

        dense_partial<<<blocks,threads,threads*sizeof(float)>>>
            (d_pool,W,d_partial,len);
        dense_final<<<1,threads,threads*sizeof(float)>>>
            (d_partial,b,d_pred,blocks);

        cudaFree(d_partial);

        lossParallel<<<1,1>>>(d_pred,label,d_loss);

        float loss,pred;
        cudaMemcpy(&loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(&pred,d_pred,sizeof(float),cudaMemcpyDeviceToHost);

        TL2 += loss;
        if((pred>0.5)==label) TA2++;

        cudaFree(d_img); cudaFree(d_norm);
        cudaFree(d_conv); cudaFree(d_pool);
        stbi_image_free(img);
    }

    printf("\nTest Accuracy: %.4f — Test Loss: %.4f\n",
           TA2/testN, TL2/testN);

    // ---------------------- FINAL GPU TIME ----------------------
    printf("\n==== FINAL GPU Kernel Time Summary ====\n");
    printf("normalize: %.3f ms\n", tNormalize);
    printf("conv     : %.3f ms\n", tConv);
    printf("pool     : %.3f ms\n", tPool);
    printf("dense    : %.3f ms\n", tDense);
    printf("loss     : %.3f ms\n", tLoss);
    printf("adam     : %.3f ms\n", tAdam);

    float totalGPU = tNormalize+tConv+tPool+tDense+tLoss+tAdam;

    printf("Total GPU kernel time: %.3f ms (%.3f s)\n",
           totalGPU, totalGPU/1000.0f);

    clock_t progEnd = clock();
    printf("Total program time (CPU wall clock): %.3f s\n",
           (double)(progEnd-progStart)/CLOCKS_PER_SEC);

    return 0;
}
