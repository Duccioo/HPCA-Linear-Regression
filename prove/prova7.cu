#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000

// Funzione per generare casualmente i dati
void generate_data(float *x, float *y, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        x[i] = (float) rand() / RAND_MAX;
        y[i] = 3 * x[i] + 1 + ((float) rand() / RAND_MAX - 0.5) * 0.1;
    }
}

// Funzione per calcolare la regressione lineare sulla CPU
void linear_regression_cpu(float *x, float *y, float *w, int n, int epochs, float lr) {
    float b = 0.0, m = 0.0;
    for (int e = 0; e < epochs; e++) {
        float b_grad = 0.0, m_grad = 0.0;
        for (int i = 0; i < n; i++) {
            float y_pred = b + m * x[i];
            b_grad += (y_pred - y[i]);
            m_grad += (y_pred - y[i]) * x[i];
        }
        b -= lr * (b_grad / n);
        m -= lr * (m_grad / n);
    }
    w[0] = b;
    w[1] = m;
}

// Funzione per calcolare la regressione lineare sulla GPU
__global__ void linear_regression_gpu(float *x, float *y, float *w, int n, int epochs, float lr) {
    float b = 0.0, m = 0.0;
    for (int e = 0; e < epochs; e++) {
        float b_grad = 0.0, m_grad = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            float y_pred = b + m * x[i];
            b_grad += (y_pred - y[i]);
            m_grad += (y_pred - y[i]) * x[i];
        }
        atomicAdd(&w[0], -lr * (b_grad / n));
        atomicAdd(&w[1], -lr * (m_grad / n));
    }
}

int main(int argc, char **argv) {
    // Parsing dei parametri
    if (argc != 5) {
        printf("Usage: ./linear_regression <num_examples> <num_epochs> <learning_rate> <threads_per_block>\n");
        return 1;
    }
    int n = atoi(argv[1]);
    int epochs = atoi(argv[2]);
    float lr = atof(argv[3]);
    int threads_per_block = atoi(argv[4]);
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Allocazione della memoria
    float *x = (float *) malloc(n * sizeof(float));
    float *y = (float *) malloc(n * sizeof(float));
    float *w_cpu = (float *) malloc(2 * sizeof(float));
    float *w_gpu = (float *) malloc(2 * sizeof(float));
    float *d_x, *d_y, *d_w;
    cudaMalloc((void **) &d_x, n * sizeof(float));
    cudaMalloc((void **) &d_y, n * sizeof(float));
    cudaMalloc((void **) &d_w, 2 * sizeof(float));
    
    // Generazione dei dati
    generate_data(x, y, n);
    
    // Calcolo della regressione lineare sulla CPU
    clock_t start_cpu = clock();
    linear_regression_cpu(x, y, w_cpu, n, epochs, lr);
    clock_t end_cpu = clock();
    double time_cpu = (double) (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    
    // Copia dei dati sulla GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_gpu, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calcolo della regressione lineare sulla GPU
    clock_t start_gpu = clock();
    linear_regression_gpu<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_w, n, epochs, lr);
    cudaDeviceSynchronize();
    clock_t end_gpu = clock();
    double time_gpu = (double) (end_gpu - start_gpu) / CLOCKS_PER_SEC;
    
    // Copia dei dati dalla GPU
    cudaMemcpy(w_gpu, d_w, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Stampa dei risultati
    printf("Bias (CPU): %f\n", w_cpu[0]);
    printf("Pendenza (CPU): %f\n\n", w_cpu[1]);
    printf("Bias (GPU): %f\n", w_gpu[0]);
    printf("Pendenza (GPU): %f\n\n", w_gpu[1]);
    printf("Tempo CPU: %f\n", time_cpu);
    printf("Tempo GPU: %f\n", time_gpu);
    
    // Liberazione della memoria
    free(x);
    free(y);
    free(w_cpu);
    free(w_gpu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    
    return 0;
}    