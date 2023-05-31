#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 256

void data_gen(float *data_x, float *data_y, int numValues);

double getTime(void);

__global__ void compute_regression(float *x, float *y, float *a, float *b, int dataPoints) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dataPoints) {
        float sumX = 0;   // sum of x_i
        float sumX2 = 0;  // sum of (x_i)^2
        float sumXY = 0;  // sum of (x_i * y_i)
        float sumY = 0;   // sum of y_i
        float meanX = 0;  // mean value of x
        float meanY = 0;  // mean value of y

        for (int j = 0; j <= i; j++) {
            sumX += x[j];
            sumX2 += x[j] * x[j];
            sumXY += x[j] * y[j];
            sumY += y[j];

            meanX = sumX / (j + 1);
            meanY = sumY / (j + 1);

            if (j == 0) {
                b[i] = 0;  // cannot compute slope for first point, set to 0 instead
            } else {
                b[i] = (sumXY - sumX * meanY) / (sumX2 - sumX * meanX);  // compute slope
            }

            a[i] = meanY - b[i] * meanX;  // compute intercept
        }

    }
}

int main() {

    const int dataPoints = 1000000;  // total number of data points, should be a multiple of 4

    // int i;                           // loop counter
    double startTime, timeDuration;  // timer values

    // allocate memory for input data and coefficients
    float *x, *y, *a, *b;
    cudaMallocManaged(&x, dataPoints * sizeof(float));
    cudaMallocManaged(&y, dataPoints * sizeof(float));
    cudaMallocManaged(&a, dataPoints * sizeof(float));
    cudaMallocManaged(&b, dataPoints * sizeof(float));

    // generating input data
    printf("Generating input data...\n");
    data_gen(x, y, dataPoints);


    int numBlocks = (dataPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // compute regression
    printf("Computing on regression...\n");

    startTime = getTime();

    compute_regression<<<numBlocks, BLOCK_SIZE>>>(x, y, a, b, dataPoints);

    timeDuration = getTime() - startTime;
    cudaDeviceSynchronize();

    printf("compute time: %g s\n", timeDuration);

    // deallocate memory
    cudaFree(x);
    cudaFree(y);
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
		data_y[i] = 0.5  * (float)i - 27 + (float)rand()/RAND_MAX - 0.5; // linear function with some random variations
	}
}

//time measurement
double getTime(void) {
   struct timeval time;
   gettimeofday(&time, NULL);
   return time.tv_sec + 1e-6 * time.tv_usec;
}
