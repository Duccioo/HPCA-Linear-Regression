#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*
This is a CUDA kernel function named computeRegression that performs linear 
regression on the given dev_x and dev_y arrays of size dataPoints and stores
 the result in dev_a and dev_b arrays.

The function uses shared memory to store subsets of dev_x and dev_y arrays for
 each block of threads. The size of the shared memory is 1024, which is the 
 maximum size of a block of threads. The i index is computed using the thread 
 index and the block index, and is used to iterate through the input arrays.

The function then computes the regression parameters dev_a and dev_b using the
 shared memory data. It uses a nested loop to compute the sums of x, y, x^2, 
 and xy for each subset of the input arrays. The loop is executed until the 
 current thread index tid is greater than i. The sums are then used to compute 
 the mean values of x and y. The regression parameters are computed using 
 these mean values and the sums of x, y, x^2, and xy.

Finally, the function stores the computed regression parameters in the dev_a
 and dev_b arrays.

Overall, this function uses shared memory to optimize memory access and 
parallelism, and computes the regression parameters using nested loops.


*/

__global__ void computeRegression(float *dev_x, float *dev_y, float *dev_a, float *dev_b, int dataPoints) {
    __shared__ float shared_x[1024];
    __shared__ float shared_y[1024];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i < dataPoints) {
        float sumX = 0;
        float sumX2 = 0;
        float sumXY = 0;
        float sumY = 0;
        float meanX = 0;
        float meanY = 0;

        int tid = threadIdx.x;

        for (int j = 0; j <= i; j += blockDim.x) {
            if (tid + j <= i) {
                shared_x[tid] = dev_x[tid + j];
                shared_y[tid] = dev_y[tid + j];
            } else {
                shared_x[tid] = 0;
                shared_y[tid] = 0;
            }
            __syncthreads();

            for (int k = 0; k <= tid; k++) {
                sumX += shared_x[k];
                sumX2 += shared_x[k] * shared_x[k];
                sumXY += shared_x[k] * shared_y[k];
                sumY += shared_y[k];
            }
            __syncthreads();
        }

        if (i % blockDim.x == 0) {
            meanX = sumX / (i + 1);
            meanY = sumY / (i + 1);
        }

        dev_b[i] = (sumXY - sumX * meanY) / (sumX2 - sumX * meanX);
        dev_a[i] = meanY - dev_b[i] * meanX;
    }
    
}

void data_gen(float * data_x, float * data_y, int numValues); 

double getTime(void);

int main(){
    

    const int dataPoints = 10240;
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (dataPoints + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nMalloc");

    float *x = (float*)malloc(dataPoints * sizeof(float));
    float *y = (float*)malloc(dataPoints * sizeof(float));
    float *a = (float*)malloc(dataPoints * sizeof(float));
    float *b = (float*)malloc(dataPoints * sizeof(float));

    printf("\nMemory occuped: %lu MB",(dataPoints * sizeof(float) / (1024 * 1024) )*2);

    float *dev_x, *dev_y, *dev_a, *dev_b;

    cudaError_t err_dev_x = cudaMalloc((void**)&dev_x, dataPoints * sizeof(float));
    cudaError_t err_dev_y = cudaMalloc((void**)&dev_y, dataPoints * sizeof(float));
    cudaError_t err_dev_a = cudaMalloc((void**)&dev_a, dataPoints * sizeof(float));
    cudaError_t err_dev_b = cudaMalloc((void**)&dev_b, dataPoints * sizeof(float));

    if (err_dev_x != cudaSuccess) {
        if (err_dev_x == cudaErrorMemoryAllocation) {
            printf("Error: Ran out of memory on the GPU!\n");
            return -1;
        } else {
            printf("Error: Failed to allocate memory on the GPU!\n");
            return -1;
        }

    }
    if (err_dev_y != cudaSuccess) {
        if (err_dev_y == cudaErrorMemoryAllocation) {
            printf("Error: Ran out of memory on the GPU!\n");
            return -1;
        } else {
            printf("Error: Failed to allocate memory on the GPU!\n");
            return -1;
        }

    }
    if (err_dev_a != cudaSuccess) {
        if (err_dev_a == cudaErrorMemoryAllocation) {
            printf("Error: Ran out of memory on the GPU!\n");
            return -1;
        } else {
            printf("Error: Failed to allocate memory on the GPU!\n");
            return -1;
        }

    }
    if (err_dev_b != cudaSuccess) {
        if (err_dev_b == cudaErrorMemoryAllocation) {
            printf("Error: Ran out of memory on the GPU!\n");
            return -1;
        } else {
            printf("Error: Failed to allocate memory on the GPU!\n");
            return -1;
        }

    }

    printf("\nGenerating input data...\n");
    data_gen(x, y, dataPoints);

    cudaMemcpy(dev_x, x, dataPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, dataPoints * sizeof(float), cudaMemcpyHostToDevice);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("\nMemory occupied in GPU: %lu MB / %lu\n", (total_mem - free_mem) / (1024 * 1024),total_mem / (1024 * 1024));



    printf("\nSTARTING COMPUTATION:");
    double startTime = getTime();

    computeRegression<<<blocksPerGrid, threadsPerBlock>>>(dev_x, dev_y, dev_a, dev_b, dataPoints);

    cudaDeviceSynchronize();

    double timeDuration = getTime() - startTime;
    printf("\nEND COMPUTATION:");

    printf("\nCompute time: %g s\n", timeDuration);

    printf("\nSize of dev_a: %lu",sizeof(dev_a));

    cudaMemcpy(a, dev_a, dataPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, dataPoints * sizeof(float), cudaMemcpyDeviceToHost);

    //  Output Results
    for (int i = 0; i < dataPoints; i++) {
        printf("Regression[%d]: y = %f + %f x\n", i, a[i], b[i]);
    } 

    free(x);
    free(y);
    free(a);
    free(b);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}


// generate input data for regression
void data_gen(float * data_x, float * data_y, int numValues){ 
	printf("Input data is a linear function: 0.5 *x - 27 with randon noise added.\n");
	int i;
	for(i = 0; i<numValues; i++){
		data_x[i] = (float)i;
		data_y[i] = 0.5  * (float)i - 27 + (float)rand()/RAND_MAX - 0.5; // linear function with some random variations
	}
}

//time measurement
double getTime(void) {
   struct timeval time;
   gettimeofday(&time, NULL);
   return time.tv_sec + 1e-6 * time.tv_usec;
}
