#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define NUM_EXAMPLES 1000
#define NUM_EPOCHS 10000
#define LEARNING_RATE 0.1
#define NUM_THREADS_PER_BLOCK 32

__global__ void regression_kernel(float *x, float *y, float *theta0, float *theta1, float alpha, int num_examples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_examples) {
        float prediction = (*theta1) * x[tid] + (*theta0);
        float error = prediction - y[tid];

        atomicAdd(theta0, -alpha * error);
        atomicAdd(theta1, -alpha * error * x[tid]);
    }
}

void regression_cpu(float *x, float *y, float *theta0, float *theta1, float alpha, int num_examples, int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float theta0_gradient = 0.0, theta1_gradient = 0.0;

        for (int i = 0; i < num_examples; i++) {
            float prediction = (*theta1) * x[i] + (*theta0);
            float error = prediction - y[i];

            theta0_gradient += error;
            theta1_gradient += error * x[i];
        }

        theta0_gradient /= num_examples;
        theta1_gradient /= num_examples;

        (*theta0) -= alpha * theta0_gradient;
        (*theta1) -= alpha * theta1_gradient;
    }
}

int main(int argc, char *argv[]) {
    int num_examples = NUM_EXAMPLES;
    int num_epochs = NUM_EPOCHS;
    float alpha = LEARNING_RATE;
    int num_threads_per_block = NUM_THREADS_PER_BLOCK;

    float *x = (float *) malloc(num_examples * sizeof(float));
    float *y = (float *) malloc(num_examples * sizeof(float));
    float *theta0 = (float *) malloc(sizeof(float));
    float *theta1 = (float *) malloc(sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < num_examples; i++) {
        x[i] = (float) rand() / (float) RAND_MAX;
        y[i] = 2.0 * x[i] + 1.0;
    }

    (*theta0) = 0.0;
    (*theta1) = 0.0;

    clock_t start_cpu, end_cpu;
    float time_cpu;

    start_cpu = clock();
    regression_cpu(x, y, theta0, theta1, alpha, num_examples, num_epochs);
    end_cpu = clock();

    time_cpu = ((float) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    printf("CPU execution time: %f seconds\n", time_cpu);
    printf("Bias (theta0) (CPU): %f\n", *theta0);
    printf("Slope (theta1) (CPU): %f\n", *theta1);
    float *x_dev, *y_dev, *theta0_dev, *theta1_dev;

    cudaMalloc((void **) &x_dev, num_examples * sizeof(float));
    cudaMalloc((void **) &y_dev, num_examples * sizeof(float));
    cudaMalloc((void **) &theta0_dev, sizeof(float));
    cudaMalloc((void **) &theta1_dev, sizeof(float));

    cudaMemcpy(x_dev, x, num_examples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y, num_examples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(theta0_dev, theta0, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(theta1_dev, theta1, sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = (num_examples + num_threads_per_block - 1) / num_threads_per_block;

    clock_t start_gpu, end_gpu;
    float time_gpu;

    start_gpu = clock();

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        regression_kernel<<<num_blocks, num_threads_per_block>>>(x_dev, y_dev, theta0_dev, theta1_dev, alpha, num_examples);
    }

    cudaMemcpy(theta0, theta0_dev, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(theta1, theta1_dev, sizeof(float), cudaMemcpyDeviceToHost);

    end_gpu = clock();

    time_gpu = ((float) (end_gpu - start_gpu)) / CLOCKS_PER_SEC;

    printf("GPU execution time: %f seconds\n", time_gpu);
    printf("Bias (theta0) (GPU): %f\n", *theta0);
    printf("Slope (theta1) (GPU): %f\n", *theta1);

    free(x);
    free(y);
    free(theta0);
    free(theta1);

    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(theta0_dev);
    cudaFree(theta1_dev);

    return 0;

}

