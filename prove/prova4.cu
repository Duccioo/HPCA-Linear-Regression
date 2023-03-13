#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128



// Funzione per generare numeri casuali nell'intervallo [0, 1]
float rand_float()
{
    srand(42);
    return (float)rand() / RAND_MAX;
}

// Funzione per calcolare la somma degli elementi di un array
float sum(float *arr, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++) {
        s += arr[i];
    }
    return s;
}

// Kernel per l'aggiornamento dei pesi
__global__ void update_weights(float *X, float *y, float *w, float lr, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float y_pred = w[0] + w[1] * X[tid];
        float error = y_pred - y[tid];
        w[0] -= lr * error;
        w[1] -= lr * error * X[tid];
    }
}

// Funzione per eseguire la regressione lineare sulla CPU
void linear_regression_cpu(float *X, float *y, int n, int epochs, float lr, float *w)
{
    for (int i = 0; i < epochs; i++) {
        float y_pred[n];
        for (int j = 0; j < n; j++) {
            y_pred[j] = w[0] + w[1] * X[j];
        }
        float error[n];
        for (int j = 0; j < n; j++) {
            error[j] = y_pred[j] - y[j];
        }
        float grad0 = sum(error, n);
        float grad1 = 0;
        for (int j = 0; j < n; j++) {
            grad1 += error[j] * X[j];
        }
        w[0] -= lr * grad0;
        w[1] -= lr * grad1;
        printf("error: %f\n", error[0]);
    }
}

// Funzione per eseguire la regressione lineare sulla GPU
void linear_regression_gpu(float *X, float *y, int n, int epochs, float lr, float *w)
{
    float *d_X, *d_y, *d_w;
    cudaMalloc(&d_X, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_w, 2 * sizeof(float));
    cudaMemcpy(d_X, X, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, 2 * sizeof(float), cudaMemcpyHostToDevice);
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < epochs; i++) {
        update_weights<<<num_blocks, BLOCK_SIZE>>>(d_X, d_y, d_w, lr, n);
    }
    cudaMemcpy(w, d_w, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
}

// Funzione per stampare il modello
void print_model(float *w)
{
    printf("Bias: %f\n", w[0]);
    printf("Pendenza: %f\n", w[1]);
}
    
int main(int argc, char **argv)
{
    if (argc != 6) {
        printf("Usage: %s num_examples num_epochs learning_rate num_threads_per_block metric\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int epochs = atoi(argv[2]);
    float lr = atof(argv[3]);
    int num_threads_per_block = atoi(argv[4]);
    char *metric = argv[5];

    // Generazione dei dati
    float X[n], y[n];
    for (int i = 0; i < n; i++) {
        X[i] = i;
        y[i] = 2 * X[i] + rand_float() * 0.1;
    }

    // Inizializzazione dei pesi
    float w[2] = {rand_float(), rand_float() };
    printf("w: %f\n", w[1]);

    // Esecuzione della regressione lineare sulla CPU
    clock_t start_cpu = clock();
    linear_regression_cpu(X, y, n, epochs, lr, w);
    clock_t end_cpu = clock();
    double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    print_model(w);

    // Esecuzione della regressione lineare sulla GPU
    clock_t start_gpu = clock();
    linear_regression_gpu(X, y, n, epochs, lr, w);
    clock_t end_gpu = clock();
    double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    // Stampa del modello
    print_model(w);

    // Calcolo della metrica
    float mse = 0;
    for (int i = 0; i < n; i++) {
        float y_pred = w[0] + w[1] * X[i];
        mse += pow(y_pred - y[i], 2);
    }
    mse /= n;

    // Stampa del tempo di esecuzione e della metrica
    printf("Tempo di esecuzione sulla CPU: %f secondi\n", time_cpu);
    printf("Tempo di esecuzione sulla GPU: %f secondi\n", time_gpu);
    if (strcmp(metric, "mse") == 0) {
        printf("Mean Squared Error: %f\n", mse);
    } else if (strcmp(metric, "rmse") == 0) {
        printf("Root Mean Squared Error: %f\n", sqrt(mse));
    } else {
        printf("Metrica non valida\n");
    }

    return 0;
}