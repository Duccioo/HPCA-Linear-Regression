#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

// Funzione per generare dati casuali per la regressione lineare
void generate_data(float *x, float *y, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = 3 * x[i] + 2 + 0.1 * ((float)rand() / RAND_MAX);
    }
}

// Funzione per calcolare il costo del modello
__global__ void compute_cost(float *x, float *y, float *theta, float *cost, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y_pred = theta[0] + theta[1] * x[i];
        float error = y_pred - y[i];
        atomicAdd(cost, error * error);
    }
}

// Funzione per calcolare il gradiente del modello
__global__ void compute_gradient(float *x, float *y, float *theta, float *gradient, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y_pred = theta[0] + theta[1] * x[i];
        float error = y_pred - y[i];
        atomicAdd(&gradient[0], error);
        atomicAdd(&gradient[1], error * x[i]);
    }
}

// Funzione per aggiornare i parametri del modello
__global__ void update_parameters(float *theta, float *gradient, float learning_rate) {
    theta[0] -= learning_rate * gradient[0];
    theta[1] -= learning_rate * gradient[1];
}

// Funzione per eseguire la regressione lineare sulla CPU
void linear_regression_cpu(float *x, float *y, float *theta, int n, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float cost = 0;
        float gradient[2] = {0, 0};
        for (int i = 0; i < n; i++) {
            float y_pred = theta[0] + theta[1] * x[i];
            float error = y_pred - y[i];
            cost += error * error;
            gradient[0] += error;
            gradient[1] += error * x[i];
        }
        cost /= n;
        gradient[0] /= n;
        gradient[1] /= n;
        theta[0] -= learning_rate * gradient[0];
        theta[1] -= learning_rate * gradient[1];
    }
}
    
int main(int argc, char **argv) {
    // Parsing degli argomenti
    if (argc != 5) {
        printf("Usage: %s num_examples num_epochs learning_rate threads_per_block\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int epochs = atoi(argv[2]);
    float learning_rate = atof(argv[3]);
    int threads_per_block = atoi(argv[4]);
    // Allocazione della memoria sulla CPU
    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));
    float *theta_cpu = (float*)malloc(2 * sizeof(float));
    float *theta_gpu = (float*)malloc(2 * sizeof(float));

    // Generazione dei dati casuali
    generate_data(x, y, n);

    // Regressione lineare sulla CPU
    clock_t start_cpu = clock();
    linear_regression_cpu(x, y, theta_cpu, n, epochs, learning_rate);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    // Allocazione della memoria sulla GPU
    float *x_gpu, *y_gpu, *theta_gpu_dev, *cost_dev, *gradient_dev;
    cudaMalloc(&x_gpu, n * sizeof(float));
    cudaMalloc(&y_gpu, n * sizeof(float));
    cudaMalloc(&theta_gpu_dev, 2 * sizeof(float));
    cudaMalloc(&cost_dev, sizeof(float));
    cudaMalloc(&gradient_dev, 2 * sizeof(float));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(theta_gpu_dev, theta_gpu, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Regressione lineare sulla GPU
    dim3 dimGrid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);
    clock_t start_gpu = clock();
    for (int epoch = 0; epoch < epochs; epoch++) {
        cudaMemset(cost_dev, 0, sizeof(float));
        cudaMemset(gradient_dev, 0, 2 * sizeof(float));
        compute_cost<<<dimGrid, dimBlock>>>(x_gpu, y_gpu, theta_gpu_dev, cost_dev, n);
        compute_gradient<<<dimGrid, dimBlock>>>(x_gpu, y_gpu, theta_gpu_dev, gradient_dev, n);
        update_parameters<<<1, 1>>>(theta_gpu_dev, gradient_dev, learning_rate);
    }
    cudaMemcpy(theta_gpu, theta_gpu_dev, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end_gpu = clock();
    double gpu_time = ((double)(end_gpu - start_gpu)) / CLOCKS_PER_SEC;

    // Stampa dei risultati
    printf("CPU time: %f seconds\n", cpu_time);
    printf("GPU time: %f seconds\n", gpu_time);
    printf("CPU bias: %f\n", theta_cpu[0]);
    printf("GPU bias: %f\n", theta_gpu[0]);
    printf("CPU slope: %f\n", theta_cpu[1]);
    printf("GPU slope: %f\n", theta_gpu[1]);
    printf("Real bias: %f\n", 2.);
    printf("Real slope: %f\n", 3.);

    // Liberazione della memoria
    free(x);
    free(y);
    free(theta_cpu);
    free(theta_gpu);
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(theta_gpu_dev);
    cudaFree(cost_dev);
    cudaFree(gradient_dev);

    
    return 0;
}