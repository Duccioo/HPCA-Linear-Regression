#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_FEATURES 2
#define NUM_SAMPLES 10
#define LEARNING_RATE 0.01
#define NUM_ITERATIONS 100000

__global__ void linear_regression(float *d_X, float *d_y, float *d_theta, int m, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0;
        for (int j = 0; j < m; j++) {
            sum += d_X[j * n + i] * d_theta[j];
        }
        d_theta[m] -= alpha * (sum - d_y[i]);
        for (int j = 0; j < m; j++) {
            d_theta[j] -= alpha * (sum - d_y[i]) * d_X[j * n + i];
        }
    }
}

int main() {
    float X[NUM_FEATURES * NUM_SAMPLES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    float y[NUM_SAMPLES] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float theta[NUM_FEATURES + 1] = {0, 0, 0};
    float *d_X, *d_y, *d_theta;
    cudaMalloc((void **)&d_X, NUM_FEATURES * NUM_SAMPLES * sizeof(float));
    cudaMalloc((void **)&d_y, NUM_SAMPLES * sizeof(float));
    cudaMalloc((void **)&d_theta, (NUM_FEATURES + 1) * sizeof(float));
    cudaMemcpy(d_X, X, NUM_FEATURES * NUM_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, NUM_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, (NUM_FEATURES + 1) * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (NUM_SAMPLES + blockSize - 1) / blockSize;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        linear_regression<<<numBlocks, blockSize>>>(d_X, d_y, d_theta, NUM_FEATURES, NUM_SAMPLES, LEARNING_RATE);
    }
    cudaMemcpy(theta, d_theta, (NUM_FEATURES + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Theta0 = %f\n", theta[0]);
    printf("Theta1 = %f\n", theta[1]);
    printf("Theta2 = %f\n", theta[2]);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    return 0;
}
