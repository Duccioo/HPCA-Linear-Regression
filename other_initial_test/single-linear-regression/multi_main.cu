#include <iostream>
#include <cuda.h>
#include <cstdint>
#include <array>
#include <initializer_list>
#include <time.h>
#include <chrono>
#include <cmath>
#include <limits>

#include <fstream>
#include <string>
#include <tuple>
#include <sstream>

#include "linear_regression.cuh"

#define INPUT_SIZE 100000
#define ERROR_DIMENSIONS 3
#define NUM_OF_THREADS 32
// #define MAX_J_ERROR 0.0202
#define MAX_J_ERROR 0.025
#define LEARNING_RATE 0.00001
#define MAX_ITER 50000
#define NUM_REP 10

auto total_cpu_results_update = std::chrono::high_resolution_clock::duration::zero();

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
        // std::cout << "x: " << x_val << " - y: " << y_val << std::endl;
        x[i] = x_val;
        // std::cout << "\nTYPE:" << typeid(x[i]).name() << '\n';
        y[i] = y_val;
        if (x_val == 1.0){
            // printf("\nEOF\n");
            break;
        }
        i++;
    }

    printf("Closing the file \n");
    // Close file
    file.close();
}
 

std::tuple<float,float,int> linear_regression_cpu(const std::array<float, INPUT_SIZE> &x, const std::array<float, INPUT_SIZE> &y, float intercept, float slope) {

    float j_error = std::numeric_limits<float>::max();

    // float learning_rate = LEARNING_RATE;
    int number_of_iteration_cpu = 0;

    // Start timers
    auto start_cpu_execution = std::chrono::high_resolution_clock::now();

    while(j_error > MAX_J_ERROR) {
    
        number_of_iteration_cpu++;

        // std::cout<<number_of_iteration_cpu<<std::endl;
        if (number_of_iteration_cpu > MAX_ITER){
            std::cout<<"\nMAX ITER - J-Error: "<<j_error<<std::endl;
            break;
        }
    
        //array for storing intermediate error levels
        float errors[3] = {0, 0, 0};

        for (int i = 0; i < INPUT_SIZE; ++i) {
            // Predict output based on current intercept and slope
            float y_pred = intercept + slope * x[i];
            // Calculate J for this specific index and store in errors index 0
            errors[0] += 0.5f * pow((y[i] - y_pred), 2);
            // Calculate intercept error for this index and store in errors index
            errors[1] += -(y[i] - y_pred);
            // Calculate slope error for this index
            errors[2] += -(y[i] - y_pred)*x[i];
        }

        // Normalize J based on the number of examples
        errors[0] = errors[0] / INPUT_SIZE;
        
        auto start_results_update_while = std::chrono::high_resolution_clock::now();

        // Update intercept and slope based on errors
        float intercept_new = intercept - LEARNING_RATE * errors[1];
        float slope_new = slope - LEARNING_RATE * errors[2];

        // Update
        intercept = intercept_new;
        slope = slope_new;
        j_error = errors[0];
        auto end_results_update_while = std::chrono::high_resolution_clock::now();
        total_cpu_results_update += end_results_update_while - start_results_update_while;

        // std::cout<< "J :" << j_error<< std::endl;
    }

    return {intercept,slope,number_of_iteration_cpu};
}

int main(int argc, char **argv)
{
    for (int i=0;i<NUM_REP;i++){
    std::ofstream savefile;

    std::ostringstream file_path;
    file_path<<INPUT_SIZE<<"_save.txt";
    std::string path_save = file_path.str();
    // apertura del file in modalità "app"
    savefile.open(path_save, std::ios_base::app);
       // verifica che il file sia stato aperto correttamente
    if (!savefile.is_open()) {
        std::cerr << "Impossibile aprire il file." << std::endl;
        return 1;
    }
 
    std::cout<<"Starting...\n";
    std::cout<<"\nParameteres:\n";
    std::cout<<"Input size: \t\t"<<INPUT_SIZE<<std::endl;
    std::cout<<"Number of threads: \t"<<NUM_OF_THREADS<<std::endl;
    //Define number of blocks
    long int numBlocks = (INPUT_SIZE + NUM_OF_THREADS - 1) / NUM_OF_THREADS;
    std::cout<<"Number Block:\t\t"<< numBlocks << std::endl;
    std::cout<<"Max error: \t\t"<<MAX_J_ERROR<<std::endl;
    std::cout<<"Learning rate: \t\t"<<LEARNING_RATE<<std::endl;
    
    // Total error
    float j_error = std::numeric_limits<float>::max();

    // Determine size of the x and y arrays
    size_t input_size = INPUT_SIZE * sizeof(float);
    size_t error_size = ERROR_DIMENSIONS * sizeof(float);

    // Define the pointers to the x and y arrays with their respective size reserved

    auto begin_cpu_allocate = std::chrono::high_resolution_clock::now();

    float* h_x = (float*)malloc(input_size);
    float* h_y = (float*)malloc(input_size);
    float* h_intercept = (float*)malloc(sizeof(float));
    float* h_slope = (float*)malloc(sizeof(float));
    float* h_results = (float*)malloc(error_size * numBlocks * sizeof(float));

    auto end_cpu_allocate = std::chrono::high_resolution_clock::now();
    auto elapsed_cpu_allocate = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_allocate - begin_cpu_allocate);
    
    std::array<float, INPUT_SIZE> x; 
    std::array<float, INPUT_SIZE> y; 
    // Select tha right dataset based in the INPUT_SIZE
    std::ostringstream oss;
    oss << "data/genereted/data_" << INPUT_SIZE << "_2_1.csv";
    std::string path = oss.str();

    load_data(path,x,y);

    // Compute random starting intercept and slope
    srand(time(NULL));
    float intercept = ((float) rand() / (RAND_MAX));
    float slope = ((float) rand() / (RAND_MAX));
    float init_intercept = intercept;
    float init_slope = slope;

    // Store the address of the x and y arrays into the pointers h_x and h_y (host_x and host_y)
    h_x = &x[0];
    h_y = &y[0];
    h_intercept = &intercept;
    h_slope = &slope;

    //Start measuring execution time of C tasks
    auto begin_cpu_run_time = std::chrono::high_resolution_clock::now();

    // EXECUTING CPU FUNCTION
    auto [intercept_cpu,slope_cpu,number_of_iteration_cpu] = linear_regression_cpu(x, y, init_intercept, init_slope);

    auto end_cpu_run_time = std::chrono::high_resolution_clock::now();
    auto elapsed_cpu_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_run_time - begin_cpu_run_time);

    // std::cout << "CPU Results: intercept = " << intercept_cpu << " and slope: " << slope_cpu << " # Iterations: " << number_of_iteration_cpu << std::endl;

    std::cout << "---------------     CPU     ------------------"<<std::endl;
    std::cout << "CPU-implementation execution time (micro s): " << elapsed_cpu_run_time.count() << std::endl;
    std::cout << "CPU-allocate time (micro s): "<<elapsed_cpu_allocate.count()<<std::endl;

    //Save data on the savefile
    savefile << elapsed_cpu_run_time.count() << "\t" << elapsed_cpu_allocate.count() << std::endl;

    std::cout << "\nCPU Results: intercept = " << intercept_cpu << " and slope: " << slope_cpu << " # Iterations: " << number_of_iteration_cpu << std::endl;

    // Allocate memory on GPU for the device_x (d_x) and device_y (d_y) of earlier calculated size
    float* d_x; float* d_y; float* d_intercept; float* d_slope; float* d_results;

    auto begin_gpu_allocate = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_x, input_size);
    cudaMalloc(&d_y, input_size);
    cudaMalloc(&d_results, error_size * numBlocks * sizeof(float));

    auto end_gpu_allocate = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu_allocate = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_allocate - begin_gpu_allocate);

    // Copy the values stored in pointer h_x and h_y into d_x and d_y
    // Transfer data from CPU memory to GPU memory.
    auto begin_gpu_copy = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, input_size, cudaMemcpyHostToDevice);

    auto end_gpu_copy = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu_copy = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_copy - begin_gpu_copy);

    // std::cout<<"Copied\n";

    // Define stepsize for updating slope and intercept.
    // float learning_rate = LEARNING_RATE;

    //Start timing the procedure

    j_error = std::numeric_limits<float>::max(); 

    int number_of_iteration_gpu = 0;

    auto total_gpu_allocate_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_kernel_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_get_results_update_do = std::chrono::high_resolution_clock::duration::zero();

    auto begin_gpu = std::chrono::high_resolution_clock::now();

    do{
        // Increase numeber of iterations
        number_of_iteration_gpu++;

        auto begin_gpu_allocate_do = std::chrono::high_resolution_clock::now();
        // ALlocate memory for the pointers to the intercept and slope
        cudaMalloc(&d_intercept, sizeof(float));
        cudaMalloc(&d_slope, sizeof(float));
        auto end_gpu_allocate_do = std::chrono::high_resolution_clock::now();
        total_gpu_allocate_do += end_gpu_allocate_do - begin_gpu_allocate_do;


        // Copy the local value of the intercept and slope to the device memory.
        auto begin_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_intercept, h_intercept, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slope, h_slope, sizeof(float), cudaMemcpyHostToDevice);
        auto end_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::now();
        total_gpu_copy_toDevice_do += end_gpu_copy_toDevice_do - begin_gpu_copy_toDevice_do;


        // Launch kernel on GPU with pointers to data in GPU memory

        auto begin_gpu_kernel = std::chrono::high_resolution_clock::now();

        simple_linear_regression<<<numBlocks,NUM_OF_THREADS>>>(d_x, d_y, d_intercept, d_slope, d_results, INPUT_SIZE);
        // Wait for all threads to return
        cudaDeviceSynchronize();

        auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
        total_gpu_kernel_do += end_gpu_kernel - begin_gpu_kernel;


        // Retrieve the GPU out value and store in host memory
        // auto begin_gpu_results_mem_copy = std::chrono::high_resolution_clock::now();
        auto begin_gpu_copy_toDevice_do2 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_results, d_results, error_size * numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        auto end_gpu_copy_toDevice_do2 = std::chrono::high_resolution_clock::now();
        total_gpu_copy_toDevice_do += end_gpu_copy_toDevice_do2 - begin_gpu_copy_toDevice_do2;

        // Check if a CUDA error has occurred.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "--------------------- Error: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        // Free memory, on the next iteration we will allocate this memory again.
        cudaFree(d_intercept);
        cudaFree(d_slope);

        float j_error = 0;
        float intercept_error = 0;
        float slope_error = 0;

        // auto end_gpu_copy_toDevice_do2 = std::chrono::high_resolution_clock::now();
        // total_gpu_copy_toDevice_do = end_gpu_copy_toDevice_do2 - begin_gpu_copy_toDevice_do2;

        // printf("\n-----------------------------------------------------");
        // printf("\nSize of h_result : %lu",sizeof(h_results));

        
        auto begin_gpu_get_results_update_do = std::chrono::high_resolution_clock::now();

        for (int i=0; i<numBlocks*3; i++){
            // printf("\nh_result [%d] = %f",i,h_results[i]);
            if (i%3 == 0){
                j_error += h_results[i];
            }
            if ((i-1)%3 == 0){
                intercept_error += h_results[i];
            }
            if ((i-2)%3 == 0){
                slope_error += h_results[i];
            }
        }

        // Update intercept and slope based on errors
        float intercept_new = intercept - LEARNING_RATE * intercept_error;
        float slope_new = slope - LEARNING_RATE * slope_error;

        // Update
        intercept = intercept_new;
        slope = slope_new;
        j_error = j_error / INPUT_SIZE;

        if (j_error < MAX_J_ERROR){
            break;
        }

        if (number_of_iteration_gpu > MAX_ITER){
            std::cout<<"\nMAX ITER - J-Error: "<<j_error<<std::endl;
            break;
        }

        auto end_gpu_get_results_update_do = std::chrono::high_resolution_clock::now();
        total_gpu_get_results_update_do += end_gpu_get_results_update_do - begin_gpu_get_results_update_do;

        // std::cout<<"numBlocks: "<<numBlocks<<" J GPU:" << j_error<< std::endl;


        // std::cout<<"\n  "<< (j_error < 1)<<std::endl;

    } while( j_error > MAX_J_ERROR);

    //End timing and compute total execution time
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu);

    //Convert time counter in micro seconds

    auto total_gpu_allocate_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_allocate_do);
    auto total_gpu_copy_toDevice_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_copy_toDevice_do);
    auto total_gpu_kernel_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_kernel_do);
    auto total_gpu_get_results_update_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_get_results_update_do);

    // Print out latest values for total error, and intercept and slope respective errors
    std::cout << "---------------     GPU     ------------------"<<std::endl;
    std::cout << "GPU-implementation execution time [TOTAL] (micro s): " << elapsed_gpu.count()  <<std::endl;
    std::cout << "                                                     " << total_gpu_allocate_do_micro.count() + total_gpu_copy_toDevice_do_micro.count() + total_gpu_kernel_do_micro.count() + total_gpu_get_results_update_do_micro.count() <<std::endl;
    std::cout << "GPU-allocate time (micro s): "<<elapsed_gpu_allocate.count()<<std::endl;
    std::cout << "GPU-copy time (micro s): "<<elapsed_gpu_copy.count()<<std::endl;
    std::cout << "GPU-allocate do_cycle time (micro s): "<<total_gpu_allocate_do_micro.count()<<std::endl;
    std::cout << "GPU-copy do_cycle time (micro s): "<<total_gpu_copy_toDevice_do_micro.count()<<std::endl;
    std::cout << "GPU-kernel do_cycle(micro s): "<<total_gpu_kernel_do_micro.count()<<std::endl;
    std::cout << "GPU-get results & update(micro s): "<<total_gpu_get_results_update_do_micro.count()<<std::endl;

    //save data on the savefile
    savefile<<elapsed_gpu.count()<<"\t"<<elapsed_gpu_allocate.count()<<"\t"<<elapsed_gpu_copy.count()<<"\t"<<total_gpu_allocate_do_micro.count()<<"\t"<<total_gpu_copy_toDevice_do_micro.count()<<"\t"<<total_gpu_kernel_do_micro.count()<<"\t"<<total_gpu_get_results_update_do_micro.count()<<std::endl;

    std::cout << "GPU Results: intercept = " << intercept << " and slope: " << slope << " # Iterations: "<< number_of_iteration_gpu <<  std::endl;


    // Free memory on GPU
    cudaFree(d_x);
    cudaFree(d_y);
}
    return 0;
}


