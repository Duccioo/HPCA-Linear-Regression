#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/***************************************************************************
*
* Filename : LinearRegressionGPU.cu
* Date: 6 Mar 2023
* 
* Description: CUDA implementation of a linear regression using least 
* squares. The goal is to find a straight line through n data points that
* minimises the sum of the square of errors (least squares).
* 
* Find a,b for y = b*x + a that minimise the error sum (b*x_i + a - y_i)^2
*
* The coefficients a and b are calulated by:
*
* b = (sum(x_i * y_i) - 1/n*(sum x_i)*(sum y_i)) / (sum(x_i^2) - 1/n*(sum x_i)^2)
* a = 1/n(sum y_i) - a/n (sum x_i)
*
* The regression is perfromed in a continous way, i.e. for each new data
* point, the regression is updated with new a,b value pair.
*
* **************************************************************************/

#define THREADS_PER_BLOCK 64


__global__ void simple_linear_regression(float* d_x, float* d_y, float* d_bias, float* d_intercept, int in_size) {
	// Define shared memory in thread block where we maintian the collective sum of squared errors
	__shared__ float errors[1];

	// Retrieve thread index to use a array index
	int index = threadIdx.x;

	// Calculate y_predicted based on current bias and intercept
	float y_pred = *d_bias + *d_intercept * d_x[index];

	// Calculate J for this specific index and store in errors index 0
	float j = 0.5f * pow((d_y[index] - y_pred), 2);
	errors[0] += atomicAdd(&errors[0], j);

	// Calculate bias error for this index and store in errors index
	errors[1] += atomicAdd(&errors[1], - (d_y[index] - y_pred));

	// Calculate intercept error for this index
	float intercept_err = -(d_y[index] - y_pred)*d_x[index];
	errors[2] += atomicAdd(&errors[2], intercept_err);

	// Wait until threads are synchronized before returning the shared memory data to host memory.
	__syncthreads();

	// Update the output values
	if (threadIdx.x == in_size - 1) {
		// Write to host memory
		d_results[0] = errors[0];
        d_results[1] = errors[1];
        d_results[2] = errors[2];


		// Reset shared memory for the next iteration of the algorithm
		errors[0] = 0;
        errors[1] = 0;
        errors[2] = 0;

	}
}

__global__ void regression(float *x, float *y, float *a, float *b,const int dataPoints) {

    

}


void data_gen(float * data_x, float * data_y, int numValues);

double getTime(void); 

int main() {
    const int dataPoints = 1000; //total number of data points, should be a multiples of 4
    int i; // loop counter
    double startTime, timeDuration; // timer values    

    // Allocare memory on host devce
    //allocate memory for input data and coefficients on the host
	float *x = (float*)calloc(dataPoints, sizeof(float)); // input data x
	float *y = (float*)calloc(dataPoints, sizeof(float)); // input data y
	float *a = (float*)calloc(dataPoints, sizeof(float)); // coefficients a (intercept)
	float *b = (float*)calloc(dataPoints, sizeof(float)); // coefficients b (slope)

    //generating input data
	printf("Generating input data...\n");
	data_gen(x, y, dataPoints);

	//allocate memory for input data and coefficients on the device
	float *d_x, *d_y, *d_a, *d_b;
	cudaMalloc(&d_x, dataPoints*sizeof(float));
	cudaMalloc(&d_y, dataPoints*sizeof(float));
	cudaMalloc(&d_a, dataPoints*sizeof(float));
	cudaMalloc(&d_b, dataPoints*sizeof(float));

	//transfer input data from host to device
	cudaMemcpy(d_x, x, dataPoints*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, dataPoints*sizeof(float), cudaMemcpyHostToDevice);

    //compute regression on the device
    printf("Computing on regression...\n");

    int numBlocks = (dataPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    startTime = getTime();

    regression<<<numBlocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_a, d_b, dataPoints);

    cudaDeviceSynchronize();

    timeDuration = getTime() - startTime;

    //transfer output data from device to host
    cudaMemcpy(a, d_a, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);

    //output Results
    for(i = dataPoints-50; i < dataPoints; i++){
        printf("regression[%d]: y = %f + %f x\n",i, a[i], b[i]);
    }

    printf("compute time: %g s\n", timeDuration);

    //deallocate memory
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFree(a);
    cudaFree(b);

    return 0;
}


// generate input data for regression
void data_gen(float * data_x, float * data_y, int numValues){ 
	printf("Input data is a linear function: 0.5 *x - 27 with randon noise added.\n");
	int i;
	for(i = 0; i<numValues; i++){
		data_x[i] = (float)i;
		data_y[i] = 0.5  * (float)i - 27 + (float)rand()/RAND_MAX - 0.5 ; // linear function with some random variations
	}
}

//time measurement
double getTime(void) {
   struct timeval time;
   gettimeofday(&time, NULL);
   return time.tv_sec + 1e-6 * time.tv_usec;
}
