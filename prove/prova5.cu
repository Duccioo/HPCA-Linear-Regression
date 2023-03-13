#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define N 1000000  // numero di esempi
#define EPOCHS 100  // numero di epoche
#define LEARNING_RATE 0.01  // learning rate
#define THREADS_PER_BLOCK 256  // numero di thread per block

// funzione per generare numeri casuali in virgola mobile tra -1 e 1
double rand_double() {
    return -1.0 + (2.0 * rand()) / RAND_MAX;
}

// kernel per il calcolo delle previsioni
__global__ void predict(double *X, double *y, double *theta, double *predictions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double prediction = 0;
        for (int j = 0; j < 2; j++) {
            prediction += X[i * 2 + j] * theta[j];
        }
        predictions[i] = prediction;
    }
}

// funzione per il calcolo dell'errore quadratico medio
double mse(double *y_true, double *y_pred) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / N;
}

int main(int argc, char **argv) {
    // parsing dei parametri
    int num_examples = N;
    int num_epochs = EPOCHS;
    double learning_rate = LEARNING_RATE;
    int threads_per_block = THREADS_PER_BLOCK;
    if (argc >= 2) num_examples = atoi(argv[1]);
    if (argc >= 3) num_epochs = atoi(argv[2]);
    if (argc >= 4) learning_rate = atof(argv[3]);
    if (argc >= 5) threads_per_block = atoi(argv[4]);

    // allocazione della memoria
    double *X = (double *) malloc(num_examples * 2 * sizeof(double));
    double *y = (double *) malloc(num_examples * sizeof(double));
    double *theta = (double *) malloc(2 * sizeof(double));
    double *predictions = (double *) malloc(num_examples * sizeof(double));
    double *d_X, *d_y, *d_theta, *d_predictions;
    cudaMalloc((void **) &d_X, num_examples * 2 * sizeof(double));
    cudaMalloc((void **) &d_y, num_examples * sizeof(double));
    cudaMalloc((void **) &d_theta, 2 * sizeof(double));
    cudaMalloc((void **) &d_predictions, num_examples * sizeof(double));

    // inizializzazione dei dati
    srand(42);
    for (int i = 0; i < num_examples; i++) {
        X[i * 2] = rand_double();
        X[i * 2 + 1] = rand_double();
        y[i] = 2 * X[i * 2] + 3 + rand_double() * 0.01;
    }
    theta[0] = rand_double();
    theta[1] = rand_double();

    // training sulla CPU
    clock_t start_cpu = clock();
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double *y_pred = (double *) malloc(num_examples * sizeof(double));
        for (int i = 0; i < num_examples; i++) {
            double prediction = 0;
            for (int j = 0; j < 2; j++) {
                prediction += X[i * 2 + j] * theta[j];
            }
            y_pred[i] = prediction;
            for (int j = 0; j < 2; j++) {
                theta[j] += learning_rate * (y[i] - prediction) * X[i * 2 + j];
            }
        }
        double error = mse(y, y_pred);
        free(y_pred);
        printf("Epoch %d - Error: %f\n", epoch, error);
    }
    clock_t end_cpu = clock();
    double time_cpu = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    
    // training sulla GPU
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        predict<<<(num_examples + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_X, d_y, d_theta, d_predictions);
        cudaMemcpy(predictions, d_predictions, num_examples * sizeof(double), cudaMemcpyDeviceToHost);
        double *y_pred = predictions;
        for (int j = 0; j < 2; j++) {
            double sum = 0;
            for (int i = 0; i < num_examples; i++) {
                sum += (y[i] - y_pred[i]) * X[i * 2 + j];
            }
            theta[j] += learning_rate * sum / num_examples;
        }
        double error = mse(y, y_pred);
        printf("Epoch %d - Error: %f\n", epoch, error);
    }
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);
    
    // stampa dei risultati
    printf("\nTraining completed\n");
    printf("Time CPU: %f seconds\n", time_cpu);
    printf("Time GPU: %f seconds\n\n", time_gpu / 1000);
    printf("MSE CPU: %f\n", mse(y, predictions));
    cudaMemcpy(d_theta, theta, 2 * sizeof(double), cudaMemcpyHostToDevice);
    predict<<<(num_examples + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_X, d_y, d_theta, d_predictions);
    cudaMemcpy(predictions, d_predictions, num_examples * sizeof(double), cudaMemcpyDeviceToHost);
    printf("MSE GPU: %f\n\n\n", mse(y, predictions));
    printf("Bias CPU: %f\n", theta[0]);
    printf("Slope CPU: %f\n\n", theta[1]);
    cudaMemcpy(d_theta, theta, 2 * sizeof(double), cudaMemcpyHostToDevice);
    predict<<<(num_examples + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_X, d_y, d_theta, d_predictions);
    cudaMemcpy(predictions, d_predictions, num_examples * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Bias GPU: %f\n", theta[0]);
    printf("Slope GPU: %f\n", theta[1]);
    
    // liberazione della memoria
    free(X);
    free(y);
    free(theta);
    free(predictions);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_predictions);
    return 0;    
}