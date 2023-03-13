#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_EXAMPLES 10000 // Numero di esempi
#define NUM_EPOCHS 1000 // Numero di epoche
#define LEARNING_RATE 0.01 // Learning rate
#define THREADS_PER_BLOCK 256 // Numero di thread per blocco

// Funzione per generare i dati casuali
void generate_data(float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = 2 * x[i] + 1 + ((float)rand() / RAND_MAX) * 0.1;
    }
}

// Funzione per calcolare la loss
__global__ void compute_loss(float* x, float* y, float* w, float* b, float* loss, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float y_pred = w[0] * x[tid] + b[0];
        loss[tid] = (y_pred - y[tid]) * (y_pred - y[tid]);
    }
}

// Funzione per aggiornare i pesi e il bias
__global__ void update_params(float* x, float* y, float* w, float* b, float lr, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float y_pred = w[0] * x[tid] + b[0];
        float error = y_pred - y[tid];
        w[0] -= lr * error * x[tid];
        b[0] -= lr * error;
    }
}

int main() {
    // Allocazione memoria sulla CPU
    float *x_cpu, *y_cpu, *w_cpu, *b_cpu, *loss_cpu;
    x_cpu = (float*)malloc(NUM_EXAMPLES * sizeof(float));
    y_cpu = (float*)malloc(NUM_EXAMPLES * sizeof(float));
    w_cpu = (float*)malloc(sizeof(float));
    b_cpu = (float*)malloc(sizeof(float));
    loss_cpu = (float*)malloc(NUM_EXAMPLES * sizeof(float));

    // Generazione dei dati
    generate_data(x_cpu, y_cpu, NUM_EXAMPLES);

    // Inizializzazione dei pesi e del bias
    w_cpu[0] = 0.0;
    b_cpu[0] = 0.0;

    // Allocazione memoria sulla GPU
    float *x_gpu, *y_gpu, *w_gpu, *b_gpu, *loss_gpu;
    cudaMalloc(&x_gpu, NUM_EXAMPLES * sizeof(float));
    cudaMalloc(&y_gpu, NUM_EXAMPLES * sizeof(float));
    cudaMalloc(&w_gpu, sizeof(float));
    cudaMalloc(&b_gpu, sizeof(float));
    cudaMalloc(&loss_gpu, NUM_EXAMPLES * sizeof(float));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(x_gpu, x_cpu, NUM_EXAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y_cpu, NUM_EXAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, w_cpu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, sizeof(float), cudaMemcpyHostToDevice);

    // Tempo di inizio per il calcolo sulla GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Esecuzione dell'algoritmo sulla GPU
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        compute_loss<<<(NUM_EXAMPLES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x_gpu, y_gpu, w_gpu, b_gpu, loss_gpu, NUM_EXAMPLES);
        update_params<<<(NUM_EXAMPLES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x_gpu, y_gpu, w_gpu, b_gpu, LEARNING_RATE, NUM_EXAMPLES);
    }

    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy(w_cpu, w_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(loss_cpu, loss_gpu, NUM_EXAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    // Calcolo della loss finale sulla CPU
    float loss_sum = 0.0;
    for (int i = 0; i < NUM_EXAMPLES; i++) {
        loss_sum += loss_cpu[i];
    }
    float loss_avg = loss_sum / NUM_EXAMPLES;

    // Tempo di fine per il calcolo sulla GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop);

    // Esecuzione dell'algoritmo sulla CPU
    float y_pred;
    float loss;
    float elapsed_time_cpu = 0.0;
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for (int i = 0; i < NUM_EXAMPLES; i++) {
            y_pred = w_cpu[0] * x_cpu[i] + b_cpu[0];
            loss = (y_pred - y_cpu[i]) * (y_pred - y_cpu[i]);
            w_cpu[0] -= LEARNING_RATE * (y_pred - y_cpu[i]) * x_cpu[i];
            b_cpu[0] -= LEARNING_RATE * (y_pred - y_cpu[i]);
        }
    }

    // Calcolo della loss finale sulla CPU
    loss_sum = 0.0;
    for (int i = 0; i < NUM_EXAMPLES; i++) {
        y_pred = w_cpu[0] * x_cpu[i] + b_cpu[0];
        loss_sum += (y_pred - y_cpu[i]) * (y_pred - y_cpu[i]);
    }
    float loss_avg_cpu = loss_sum / NUM_EXAMPLES;

    // Tempo di esecuzione sulla CPU
    printf("Tempo di esecuzione sulla CPU: %f secondi\n", elapsed_time_cpu / 1000.0);

    // Tempo di esecuzione sulla GPU
    printf("Tempo di esecuzione sulla GPU: %f secondi\n", elapsed_time_gpu / 1000.0);

    // Metrica sulla bontà del modello
    printf("Loss finale sulla CPU: %f\n", loss_avg_cpu);
    printf("Loss finale sulla GPU: %f\n", loss_avg);

    // Deallocazione memoria
    free(x_cpu);
    free(y_cpu);
    free(w_cpu);
    free(b_cpu);
    free(loss_cpu);
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(w_gpu);
    cudaFree(b_gpu);
    cudaFree(loss_gpu);

    return 0;
}
