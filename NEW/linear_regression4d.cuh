#define ERROR_DIMENSIONS 5
                        // (d_x1,d_x2,d_x3, d_y, d_intercept, d_slope1,d_slope2,d_slope3, d_results, INPUT_SIZE)
__global__ void simple_linear_regression(float* d_x1,float* d_x2,float* d_x3, float* d_y, float* d_intercept, float* d_slope1,float* d_slope2, float* d_slope3, float* d_results, int in_size) {
    // Define shared memory in thread block where we maintian the collective sum of squared errors
    __shared__ float errors[ERROR_DIMENSIONS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < in_size) {
        // Calculate y_predicted based on current bias and intercept
        float y_pred = *d_intercept + *d_slope1 * d_x1[index] + *d_slope2 * d_x2[index] + *d_slope3 * d_x3[index];

        // Calculate J for this specific index and store in errors index 0
        atomicAdd(&errors[0], 0.5f * pow((d_y[index] - y_pred), 2));

        // Calculate bias error for this index and store in errors index
        atomicAdd(&errors[1], - (d_y[index] - y_pred));

        // Calculate intercept error for this index
        atomicAdd(&errors[2], -(d_y[index] - y_pred)*d_x1[index]);

        // Calculate intercept error for this index
        atomicAdd(&errors[3], -(d_y[index] - y_pred)*d_x2[index]);

        // Calculate intercept error for this index
        atomicAdd(&errors[4], -(d_y[index] - y_pred)*d_x3[index]);

        // Wait until threads are synchronized before returning the shared memory data to host memory.
        __syncthreads();

        // Update the output values
        if (threadIdx.x == 0) {
            // Write to host memory
            d_results[(blockIdx.x * 5) + 0] = errors[0];
            d_results[(blockIdx.x * 5) + 1] = errors[1];
            d_results[(blockIdx.x * 5) + 2] = errors[2];
            d_results[(blockIdx.x * 5) + 3] = errors[3];
            d_results[(blockIdx.x * 5) + 4] = errors[4];

            // Reset shared memory for the next iteration of the algorithm
            errors[0] = 0;
            errors[1] = 0;
            errors[2] = 0;
            errors[3] = 0;
            errors[4] = 0;

        }
    }
}