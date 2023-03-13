#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Funzione per generare numeri casuali nell'intervallo [0, 1]
float rand_float()
{
    srand(42);
    return (float)rand() / RAND_MAX;
}


// Funzione che esegue la regressione lineare sulla CPU
void cpu_regression(float* x, float* y, float* theta, int m, int n, int epochs, float alpha) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float error = 0.0;
        for (int i = 0; i < m; i++) {
            float y_hat = 0.0;
            for (int j = 0; j < n; j++) {
                y_hat += theta[j] * x[i*n+j];
            }
            error += pow(y_hat - y[i], 2);
            for (int j = 0; j < n; j++) {
                theta[j] -= alpha * (y_hat - y[i]) * x[i*n+j];
            }
        }
        
        // if (epoch%5==0){
        //     printf("Epoch %d, Error: %f\n", epoch+1, error/m);
        // }
    }
}

// Funzione kernel che esegue la regressione lineare su un blocco sulla GPU
__global__ void gpu_regression_kernel(float* x, float* y, float* theta, int m, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float y_hat = 0.0;
        for (int j = 0; j < n; j++) {
            y_hat += theta[j] * x[idx*n+j];
        }
        for (int j = 0; j < n; j++) {
            atomicAdd(&theta[j], -alpha * (y_hat - y[idx]) * x[idx*n+j]);
        }
    }
}

// Funzione che esegue la regressione lineare sulla GPU
void gpu_regression(float* x, float* y, float* theta, int m, int n, int epochs, float alpha, int threads_per_block) {
    float *d_x, *d_y, *d_theta;
    cudaMalloc(&d_x, m*n*sizeof(float));
    cudaMalloc(&d_y, m*sizeof(float));
    cudaMalloc(&d_theta, n*sizeof(float));
    cudaMemcpy(d_x, x, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, n*sizeof(float), cudaMemcpyHostToDevice);
    int blocks_per_grid = ceil((float)m / threads_per_block);
    for (int epoch = 0; epoch < epochs; epoch++) {
        gpu_regression_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_theta, m, n, alpha);
        cudaDeviceSynchronize();
        float error = 0.0;
        cudaMemcpy(theta, d_theta, n*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < m; i++) {
            float y_hat = 0.0;
            for (int j = 0; j < n; j++) {
                y_hat += theta[j] * x[i*n+j];
            }
            error += pow(y_hat - y[i], 2);
        }
        // if (epoch%5==0){
        //     printf("Epoch %d, Error: %f\n", epoch+1, error/m);
        // }
    }
    cudaMemcpy(theta, d_theta, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_theta);
}
    
int main(int argc, char** argv) {
    // Leggi i parametri dall'input
    int m = atoi(argv[1]); // Numero di esempi
    int n = 2; // Numero di feature (in questo caso 2 per semplicità)
    int epochs = atoi(argv[2]); // Numero di epoche
    float alpha = atof(argv[3]); // Learning rate
    int threads_per_block = atoi(argv[4]); // Numero di thread per block

    // Genera un dataset di esempio (in questo caso una linea retta con rumore)
    float* x = (float*)malloc(m*n*sizeof(float));
    float* y = (float*)malloc(m*sizeof(float));
    srand(42);
    for (int i = 0; i < m; i++) {
        x[i*n] = 1.0; // Bias
        x[i*n+1] = (float)i / m; // Feature 1
        y[i] = 4.0 + 3.0*x[i*n+1] + 0.1*((float)rand()/RAND_MAX); // Linea retta con rumore
    }

    // Inizializza i pesi
    float* theta = (float*)malloc(n*sizeof(float));
    for (int j = 0; j < n; j++) {
        theta[j] = rand_float(); // Inizializzazione casuale
    }

    // Esegui la regressione lineare sulla CPU e sulla GPU
    printf("Regressione lineare sulla CPU\n");
    clock_t start_cpu = clock();
    cpu_regression(x, y, theta, m, n, epochs, alpha);
    clock_t end_cpu = clock();
    double time_cpu = (double) (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Bias: %f, Pendenza: %f\n", theta[0], theta[1]);


    printf("\nRegressione lineare sulla GPU\n");
    clock_t start_gpu = clock();
    gpu_regression(x, y, theta, m, n, epochs, alpha, threads_per_block);
    clock_t end_gpu = clock();
    double time_gpu = (double) (end_gpu - start_gpu) / CLOCKS_PER_SEC;
    
    printf("Bias: %f, Pendenza: %f\n", theta[0], theta[1]);

    // Libera la memoria
    free(x);
    free(y);
    free(theta);

    printf("\nTempo CPU: %f\n", time_cpu);
    printf("Tempo GPU: %f\n", time_gpu);

    return 0;
}