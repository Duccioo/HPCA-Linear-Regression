/**
 * Simple Linear Regression implementation in CUDA
 *
 * @author: Yvo Elling
 * @date: 08-01-21
 */

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cstdint>
#include <array>
#include <initializer_list>
#include <time.h>
#include <chrono>
#include <cmath>
#include <limits>

#include "linear_regression.cuh"

#define INPUT_SIZE 1000
#define THREADS_PER_BLOCK 1024
#define ERROR_DIMENSIONS 3

void linear_regression_cpu(std::array<float, INPUT_SIZE> x, std::array<float, INPUT_SIZE> y, float bias, float intercept) {

    printf("Start regression\n");
	float j_error = std::numeric_limits<float>::max();
	
	float learning_rate = 0.000000001;

	while(j_error > 5) {
        std::cout << "J_error " << j_error << std::endl;
		//array for storing intermediate error levels
		float errors[3] = {0, 0, 0};

		for (int i = 0; i < INPUT_SIZE; ++i) {
			// Predict output based on current bias and intercept
			float y_pred = bias + intercept * x[i];

			// Calculate J for this specific index and store in errors index 0
			errors[0] += 0.5f * pow((y[i] - y_pred), 2);
			// Calculate bias error for this index and store in errors index
			errors[1] += -(y[i] - y_pred);
			// Calculate intercept error for this index
			errors[2] += -(y[i] - y_pred)*x[i];
		}

		// Update bias and intercept based on errors
		float bias_new = bias - learning_rate * errors[1];
		float intercept_new = intercept - learning_rate * errors[2];

		// Update
		bias = bias_new;
		intercept = intercept_new;
		j_error = errors[0];
	}

	std::cout << "CPU Results: Bias = " << bias << " and Intercept: " << intercept << std::endl;
}

void load_data(std::string filename, std::array<float, INPUT_SIZE>& x, std::array<float, INPUT_SIZE>& y) {
    // Open file
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return;
    }

    // Read header
    std::string line;
    std::getline(file, line);

    // Read data
    int i = 0;
    while (std::getline(file, line)) {
        float x_val, y_val;
        std::sscanf(line.c_str(), "%f,%f", &x_val, &y_val);
        std::cout << "x: " << x_val << "y: " << y_val << std::endl;
        x[i] = x_val;
        y[i] = y_val;
        if (x_val == 1.0){
            printf("\nEOF\n");
            break;
        }
        i++;
    }

    printf("Closing the file \n");
    // Close file
    file.close();
}


int main(int argc, char **argv)
{
	// Total error
	float j_error = std::numeric_limits<float>::max();

	// Determine size of the x and y arrays
	size_t input_size = INPUT_SIZE * sizeof(float);
	size_t error_size = ERROR_DIMENSIONS * sizeof(float);

	// Define the pointers to the x and y arrays with their respective size reserved
	float* h_x = (float*)malloc(input_size);
	float* h_y = (float*)malloc(input_size);
	float* h_bias = (float*)malloc(sizeof(float));
	float* h_intercept = (float*)malloc(sizeof(float));
	float* h_results = (float*)malloc(error_size);

	// Initial values that are used for the linear regression
	// std::array<float, INPUT_SIZE> x = {0.00f, 0.22f, 0.24f, 0.33f, 0.37f, 0.44f, 0.44f, 0.57f, 0.93f, 1.00f};
	// std::array<float, INPUT_SIZE> y = {0.00f, 0.22f, 0.58f, 0.20f, 0.55f, 0.39f, 0.54f, 0.53f, 1.00f, 0.61f};

    std::array<float, INPUT_SIZE> x, y;
    load_data("data/genereted/data_1000_2_1.csv", x, y);
    printf("Data loaded\n");
    // return 0;
	// Compute random starting bias and intercept
	srand(time(NULL));
	float bias = ((float) rand() / (RAND_MAX));
	float intercept = ((float) rand() / (RAND_MAX));
	float init_bias = bias;
	float init_intercept = intercept;

	// Store the address of the x and y arrays into the pointers h_x and h_y (host_x and host_y)
	h_x = &x[0];
	h_y = &y[0];
	h_bias = &bias;
	h_intercept = &intercept;

	//Start measuring execution time of C tasks
	auto begin = std::chrono::high_resolution_clock::now();
	linear_regression_cpu(x, y, init_bias, init_intercept);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
	std::cout << "C-implementation execution time(ns): " << elapsed.count() << std::endl;

	// Allocate memory on GPU for the device_x (d_x) and device_y (d_y) of earlier calculated size
	float* d_x; float* d_y; float* d_bias; float* d_intercept; float* d_results;
	cudaMalloc(&d_x, input_size);
	cudaMalloc(&d_y, input_size);
	cudaMalloc(&d_results, error_size);

	// Copy the values stored in pointer h_x and h_y into d_x and d_y
	// Transfer data from CPU memory to GPU memory.
	cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, input_size, cudaMemcpyHostToDevice);

	// Define stepsize for updating intercept and bias.
	float learning_rate = 0.000000000000001;

	//Start timing the procedure
	auto begin_gpu = std::chrono::high_resolution_clock::now();
    j_error = 1800;
    return 0;
	do {

        std::cout << "j_error_gpu " << j_error<< "\nExit value: " << (j_error > 1225) <<std::endl;
		// ALlocate memory for the pointers to the bias and intercept
		cudaMalloc(&d_bias, sizeof(float));
		cudaMalloc(&d_intercept, sizeof(float));

		// Copy the local value of the bias and intercept to the device memory.
		cudaMemcpy(d_bias, h_bias, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_intercept, h_intercept, sizeof(float), cudaMemcpyHostToDevice);

        int numBlocks = (INPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		// Launch kernel on GPU with pointers to data in GPU memory
		simple_linear_regression<<<numBlocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_bias, d_intercept, d_results, INPUT_SIZE);

		// Wait for all threads to return
		cudaDeviceSynchronize();

		// Retrieve the GPU out value and store in host memory
		cudaMemcpy(h_results, d_results, error_size, cudaMemcpyDeviceToHost);

		// Check if a CUDA error has occurred.
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
			break;
		}

		// Free memory, on the next iteration we will allocate this memory again.
		cudaFree(d_bias);
		cudaFree(d_intercept);

		// Update bias and intercept based on errors
		float bias_new = bias - learning_rate * h_results[1];
		float intercept_new = intercept - learning_rate * h_results[2];

		// Update
		bias = bias_new;
		intercept = intercept_new;
		j_error = h_results[0];
        std::cout << "h_results[0] : " << h_results[0] << std::endl;
	}
    while(j_error > 12);
	//End timing and compute total execution time
	auto end_gpu = std::chrono::high_resolution_clock::now();
	auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu);

	// Print out latest values for total error, and bias and intercept respective errors
	std::cout << "GPU Results: Bias = " << bias << " and Intercept: " << intercept << std::endl;
	std::cout << "GPU-implementation execution time(ns): " << elapsed_gpu.count() << std::endl;


	// Free memory on GPU
	cudaFree(d_x);
	cudaFree(d_y);

	return 0;
}