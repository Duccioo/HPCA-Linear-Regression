#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#include "kernel.cuh"

#define NUM_SAMPLES 100
#define NUM_FEATURES 1
#define LEARNING_RATE 0.01
#define ITERATIONS 1000
#define NUM_THREADS 1024

// Funzione per generare i dati di input
void generateData(float* X, float* y, float theta0, float theta1)
{
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        X[i * NUM_FEATURES] = i*0.1;                      // Prima colonna di X è un valore incrementale
        // X[i * NUM_FEATURES + 1] = 2.0f * i + 1.0f;    // Seconda colonna di X è un valore incrementale multiplo di 2, con un offset di 1
        y[i] = theta0 * X[i * NUM_FEATURES] + theta1;  // y = theta0*x1 + theta1
    }
}


int main()
{
    // Allocazione della memoria sul dispositivo
    float* d_X;
    float* d_y;
    float* d_theta;
    float* d_predictions;
    float* d_error;
    float* d_tX;
    float* d_dw;
    float* d_db;



    cudaMalloc((void**)&d_X, NUM_SAMPLES * NUM_FEATURES * sizeof(float));
    cudaMalloc((void**)&d_tX, NUM_SAMPLES * NUM_FEATURES * sizeof(float));
    cudaMalloc((void**)&d_y, NUM_SAMPLES * sizeof(float));

    cudaMalloc((void**)&d_db, sizeof(float));
    cudaMalloc((void**)&d_dw, NUM_FEATURES *  sizeof(float));

    cudaMalloc((void**)&d_theta, (NUM_FEATURES+1) * sizeof(float));
    cudaMalloc((void**)&d_predictions, NUM_SAMPLES * sizeof(float));
    cudaMalloc((void**)&d_error, NUM_SAMPLES * sizeof(float));

    // Impostazione dei theta manualmente
    float theta[NUM_FEATURES+1] = { 5.0, 2.0 };
 

    // Generazione dei dati di input
    float X[NUM_SAMPLES * NUM_FEATURES];  // Matrice di input (100 righe x 2 colonne)
    float y[NUM_SAMPLES];                  // Vettore dei valori target
    generateData(X, y, theta[0], theta[1]);
    printf("y==%f\n",y[1]);

    // Copia dei dati di input sulla memoria del dispositivo
    cudaMemcpy(d_X, X, NUM_SAMPLES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, NUM_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);

    // Inizializzazione dei theta
    theta[0] = 0;
    theta[1] = 0;
    cudaMemcpy(d_theta, theta, (NUM_FEATURES+1) * sizeof(float), cudaMemcpyHostToDevice);

    // Esecuzione dell'algoritmo del gradient descent
    int m=0;
    int n=0;
    int k=0;
    int grid_rows=0;
    int grid_cols=0;
    int gridBlock=0;

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        // Calcolo delle predizioni: calcolo il prodotto tra input e i pesi e bias
        // int numBlocks = (NUM_SAMPLES + NUM_THREADS -1) / NUM_THREADS;
        m = NUM_SAMPLES;
        n = (NUM_FEATURES);
        k = 1;
        grid_rows = int((m + ceilf(pow(NUM_THREADS,1/2)) - 1) / ceilf(pow(NUM_THREADS,1/2)));
        grid_cols = int((k + ceilf(pow(NUM_THREADS,1/2)) - 1) / ceilf(pow(NUM_THREADS,1/2)));
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(ceilf(pow(NUM_THREADS,1/2)), ceilf(pow(NUM_THREADS,1/2)));
        matrixMultiplyWithBias<<<dimGrid, dimBlock>>>(d_X,d_theta, d_predictions, m, n, k);
       
        // Calcolo dell'errore
        grid_rows = int((m + NUM_THREADS - 1) / NUM_THREADS);
        compute_error<<<grid_rows, NUM_THREADS>>>(d_predictions, d_y, d_error, NUM_SAMPLES);

        //calcolo la transposta della information matrix:
        gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_X, d_tX, m, n);

        // calcolo il dw
        m = NUM_FEATURES;
        n = NUM_SAMPLES;
        k = 1;
        grid_rows = int((m + ceilf(pow(NUM_THREADS,1/2)) - 1) / ceilf(pow(NUM_THREADS,1/2)));
        grid_cols = int((k + ceilf(pow(NUM_THREADS,1/2)) - 1) / ceilf(pow(NUM_THREADS,1/2)));
        dim3 dimGrid2(grid_cols, grid_rows);
        dim3 dimBlock2(ceilf(pow(NUM_THREADS,1/2)), ceilf(pow(NUM_THREADS,1/2)));
        matrixMultiply<<<dimGrid2, dimBlock2>>>(d_tX, d_error, d_dw, m, n, k);

     
        // sommo l'errore
        m = NUM_SAMPLES;
        gridBlock = int((m + NUM_THREADS - 1) / NUM_THREADS);
        vectorSum<<<gridBlock, NUM_THREADS>>>(d_error, NUM_SAMPLES, d_db);

        // Aggiornamento dei theta
        float weights[NUM_FEATURES] = { 0.0f};
        float bias[1]={0.};

        float error[NUM_SAMPLES];
        float prediction[NUM_SAMPLES];



        // mi prendo i db e dw dal Device:
        cudaMemcpy(error, d_error, NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(prediction , d_predictions,NUM_SAMPLES * sizeof(float) , cudaMemcpyDeviceToHost);
        cudaMemcpy(weights, d_dw, NUM_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias, d_db, sizeof(float), cudaMemcpyDeviceToHost);



        // normalizzo il bias e faccio l'update
        for(int i = 0; i < NUM_FEATURES; i++){
            theta[i] -= (LEARNING_RATE * weights[i]) / NUM_SAMPLES;

        }
        printf("thetastart %f\n", theta[NUM_FEATURES]);

        printf("errore %f\n", error[1]);
        printf("weight %f\n", weights[1]);
        printf("predizione %f\n", prediction[1]);
        printf("bias %f\n", bias[0]);
        
        // bias[0]=0;
        // for(int i = 0; i < NUM_SAMPLES; i++){
        //     printf("%f", error[i]);
        //     bias[0] += error[i];
        // }
        // printf("bias %f\n", bias[0]);



        // normalizzo i pesi e faccio l'update
        theta[NUM_FEATURES] -= (LEARNING_RATE * bias[0])/ NUM_SAMPLES;
        
        printf("theta %f\n\n", theta[NUM_FEATURES]);


        cudaMemcpy(d_theta, theta, (NUM_FEATURES+1) * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copia dei risultati dal dispositivo alla memoria host
    cudaMemcpy(theta, d_theta, (NUM_FEATURES+1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Stampa dei theta finali
    printf("Theta0: %.2f\n", theta[0]);
    printf("Theta1: %.2f\n", theta[1]);

    // Deallocazione della memoria sul dispositivo
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_predictions);
    cudaFree(d_error);

    return 0;
}
