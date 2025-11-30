#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h> // for rand()

// Algorithm/Kernel 1: Matrix multiply (X * w + b)
__global__ void matMulKernel(const float* A, const float* B, const float* bias, float* C, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < N) {
        float sum = 0.0f;
        for(int k = 0; k < d; k++) {
            sum += A[row * d + k] * B[k];
        }
        C[row] = sum + *bias; // Changed to use device pointer directly
    }
}

// Algorithm/Kernel 2: Sigmoid (stable version)
__global__ void sigmoidKernel(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        float z = Z[idx];
        if (z >= 0.0f) {
            Z[idx] = 1.0f / (1.0f + expf(-z));
        } else {
            float ez = expf(z);
            Z[idx] = ez / (1.0f + ez);
        }
    }
}

// Algorithm/Kernel 3: Subtract vector
__global__ void vecSubKernel(const float* A, const float* B, float* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = A[idx] - B[idx];
    }
}

// Algorithm/Kernel 6: Gradient Compute (compute grad_w and grad_b)
__global__ void gradientComputekernel(
    const float* X, const float* error,
    float* grad_w, float* grad_b, int N, int d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += X[i * d + j] * error[i];
        }
        grad_w[j] = sum;
    }

    if (j == 0) {
        float sum_b = 0.0f;
        for (int i = 0; i < N; i++) {
            sum_b += error[i];
        }
        *grad_b = sum_b;
    }
}

// Algorithm/Kernel 7: Norm 2 compute
__global__ void vectorNorm2(const float* A, const float* B, float* res, int N){
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < N) {
        float diff = A[tid] - B[tid];
        temp += diff * diff;
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(res, cache[0]);
    }
}

// Kernel update weight & bias (avoid CPU copy)
__global__ void updateKernel(float* w, float* grad_w, float* bias, float grad_b, float lr, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) {
        w[j] -= lr * grad_w[j]; // grad_w already divided by N outside
    }

    if (j == 0) {
        *bias -= lr * grad_b;
    }
}

// train function
extern "C" void train(
    const float* X, const float* y,
    float lr, float epsilon, int max_iter,
    int N, int d, float* weight, float* bias
){
    float *d_X, *d_y, *d_weight, *d_weight_prev, *d_bias, *d_error, *d_pred;
    float *d_grad_w, *d_temp_norm, *d_grad_b, *d_w_diff;

    cudaMalloc(&d_X, sizeof(float)*N*d);
    cudaMalloc(&d_y, sizeof(float)*N);
    cudaMalloc(&d_weight, sizeof(float)*d);
    cudaMalloc(&d_weight_prev, sizeof(float)*d);
    cudaMalloc(&d_bias, sizeof(float));
    cudaMalloc(&d_error, sizeof(float)*N);
    cudaMalloc(&d_pred, sizeof(float)*N);
    cudaMalloc(&d_grad_w, sizeof(float)*d);
    cudaMalloc(&d_grad_b, sizeof(float));
    cudaMalloc(&d_temp_norm, sizeof(float));
    cudaMalloc(&d_w_diff, sizeof(float)*d);

    cudaMemcpy(d_X, X, sizeof(float)*N*d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Initialize weight randomly and bias = 0
    float* h_weight = (float*)malloc(sizeof(float)*d);
    for(int i = 0; i < d; i++) {
        h_weight[i] = ((float)rand() / RAND_MAX - 0.5f); // random [-0.5,0.5]
    }
    float h_bias = 0.0f;

    cudaMemcpy(d_weight, h_weight, sizeof(float)*d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &h_bias, sizeof(float), cudaMemcpyHostToDevice);

    free(h_weight);

    dim3 block(16,16);
    dim3 grid_mat((1 + block.x -1)/block.x, (N + block.y -1)/block.y);

    dim3 block1d(256);
    dim3 grid1d_N((N + block1d.x -1)/block1d.x);
    dim3 grid1d_d((d + block1d.x -1)/block1d.x);

    for(int iter = 0; iter < max_iter; ++iter){
        // store previous weight for norm check
        cudaMemcpy(d_weight_prev, d_weight, sizeof(float)*d, cudaMemcpyDeviceToDevice);

        // Xw + b
        matMulKernel<<<grid_mat, block>>>(d_X, d_weight, d_bias, d_pred, N, d);

        // sigmoid (stable)
        sigmoidKernel<<<grid1d_N, block1d>>>(d_pred, N);

        // error = pred - y
        vecSubKernel<<<grid1d_N, block1d>>>(d_pred, d_y, d_error, N);

        // compute gradient
        gradientComputekernel<<<grid1d_d, block1d>>>(d_X, d_error, d_grad_w, d_grad_b, N, d);

        float host_grad_b;
        cudaMemcpy(&host_grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);
        host_grad_b /= N; // divide bias gradient

        // divide grad_w by N on host
        float* h_grad_w = (float*)malloc(sizeof(float)*d);
        cudaMemcpy(h_grad_w, d_grad_w, sizeof(float)*d, cudaMemcpyDeviceToHost);
        for(int i = 0; i < d; i++) {
            h_grad_w[i] /= N;
        }
        cudaMemcpy(d_grad_w, h_grad_w, sizeof(float)*d, cudaMemcpyHostToDevice);
        free(h_grad_w);

        // weight update
        updateKernel<<<grid1d_d, block1d>>>(d_weight, d_grad_w, d_bias, host_grad_b, lr, d);

        // compute ||w_prev - w||
        vecSubKernel<<<grid1d_d, block1d>>>(d_weight_prev, d_weight, d_w_diff, d);

        cudaMemset(d_temp_norm, 0, sizeof(float));
        vectorNorm2<<<grid1d_d, block1d>>>(d_weight_prev, d_weight, d_temp_norm, d);

        float norm_val;
        cudaMemcpy(&norm_val, d_temp_norm, sizeof(float), cudaMemcpyDeviceToHost);

        if(norm_val <= epsilon) {
            break;
        }
    }

    cudaMemcpy(weight, d_weight, sizeof(float)*d, cudaMemcpyDeviceToHost);
    cudaMemcpy(bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weight);
    cudaFree(d_weight_prev);
    cudaFree(d_bias);
    cudaFree(d_error);
    cudaFree(d_pred);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);
    cudaFree(d_temp_norm);
    cudaFree(d_w_diff);
}
