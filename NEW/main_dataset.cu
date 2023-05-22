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

#include "linear_regression4d.cuh"

#define INPUT_SIZE 160000
#define ERROR_DIMENSIONS 5
#define NUM_OF_THREADS 32
// #define MAX_J_ERROR 0.0202
#define MAX_J_ERROR 0.01
#define LEARNING_RATE 0.000001
#define MAX_ITER 50000
#define NUM_REP 5

auto total_cpu_results_update = std::chrono::high_resolution_clock::duration::zero();

void load_data(std::string filename, std::array<float, INPUT_SIZE>& x1, std::array<float, INPUT_SIZE>& x2, 
    std::array<float, INPUT_SIZE>& x3, std::array<float, INPUT_SIZE>& y) {
// Open file
std::ifstream file(filename);

// Read header
std::string line;
std::getline(file, line);

// Read data
int i = 0;
while (std::getline(file, line)) {
float x1_val, x2_val, x3_val, y_val;
std::sscanf(line.c_str(), "%f,%f,%f,%f", &x1_val, &x2_val, &x3_val, &y_val); 
// std::cout << "data: " << i << " " << x1_val << " " << x2_val << " " << x3_val << " " << y_val << " " << "\n";
x1[i] = x1_val; 
x2[i] = x2_val;
x3[i] = x3_val;
y[i]  = y_val;
i++;
}
}


std::tuple<float,float,float,float,int> linear_regression_cpu(const std::array<float, INPUT_SIZE> &x1,
                                                              const std::array<float, INPUT_SIZE> &x2,
                                                              const std::array<float, INPUT_SIZE> &x3,
                                                              const std::array<float, INPUT_SIZE> &y, 
                                                              float intercept,
                                                              float slope1,
                                                              float slope2,
                                                              float slope3) {

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
        float errors[5] = {0, 0, 0, 0, 0};

        for (int i = 0; i < INPUT_SIZE; ++i) {
            // Predict output based on current intercept and slope
            float y_pred = intercept + slope1*x1[i] + slope2*x2[i] + slope3*x3[i];
            // Calculate J for this specific index and store in errors index 0
            errors[0] += 0.5f * pow((y[i] - y_pred), 2);
            // Calculate intercept error for this index and store in errors index
            errors[1] += -(y[i] - y_pred);
            // Calculate slope error for this index
            errors[2] += -(y[i] - y_pred)*x1[i];
            errors[3] += -(y[i] - y_pred)*x2[i];
            errors[4] += -(y[i] - y_pred)*x3[i];
        }

        // Normalize J based on the number of examples
        errors[0] = errors[0] / INPUT_SIZE;
        
        auto start_results_update_while = std::chrono::high_resolution_clock::now();

        // Update intercept and slope based on errors
        float intercept_new = intercept - LEARNING_RATE * errors[1];
        float slope_new_1 = slope1 - LEARNING_RATE * errors[2];
        float slope_new_2 = slope2 - LEARNING_RATE * errors[3];
        float slope_new_3 = slope3 - LEARNING_RATE * errors[4];

        // Update
        intercept = intercept_new;
        slope1 = slope_new_1;
        slope2 = slope_new_2;
        slope3 = slope_new_3;
        j_error = errors[0];
        auto end_results_update_while = std::chrono::high_resolution_clock::now();
        total_cpu_results_update += end_results_update_while - start_results_update_while;

        // std::cout<< "J :" << j_error<< std::endl;
    }

    return {intercept,slope1,slope2,slope3,number_of_iteration_cpu};
}

// #define INPUT_SIZE 160000
// #define ERROR_DIMENSIONS 5
// #define NUM_OF_THREADS 32
// // #define MAX_J_ERROR 0.0202
// #define MAX_J_ERROR 0.01
// #define LEARNING_RATE 0.000001
// #define MAX_ITER 50000
// #define NUM_REP 30


int main(int argc, char **argv)
{
    for (int i=0;i<NUM_REP;i++){

    std::ofstream savefile;
    std::ostringstream file_path;
    file_path<<"save/"<<NUM_REP<<"_"<<NUM_OF_THREADS<<"_"<<MAX_J_ERROR<<"_"<<LEARNING_RATE<<"_save.txt";
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
    // float j_error = std::numeric_limits<float>::max();

    // Determine size of the x and y arrays
    size_t input_size = INPUT_SIZE * sizeof(float);
    size_t error_size = ERROR_DIMENSIONS * sizeof(float);

    // Define the pointers to the x and y arrays with their respective size reserved

    auto begin_cpu_allocate = std::chrono::high_resolution_clock::now();

    float* h_x1 = (float*)malloc(input_size);
    float* h_x2 = (float*)malloc(input_size);
    float* h_x3 = (float*)malloc(input_size);
    float* h_y = (float*)malloc(input_size);
    float* h_intercept = (float*)malloc(sizeof(float));
    float* h_slope1 = (float*)malloc(sizeof(float));
    float* h_slope2 = (float*)malloc(sizeof(float));
    float* h_slope3 = (float*)malloc(sizeof(float));
    float* h_results = (float*)malloc(error_size * numBlocks * sizeof(float));

    auto end_cpu_allocate = std::chrono::high_resolution_clock::now();
    auto elapsed_cpu_allocate = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_allocate - begin_cpu_allocate);
    
    std::array<float, INPUT_SIZE> x1; 
    std::array<float, INPUT_SIZE> x2;
    std::array<float, INPUT_SIZE> x3;
    std::array<float, INPUT_SIZE> y; 

    load_data("data/output_norm.csv", x1, x2, x3, y);

    // Compute random starting intercept and slope
    srand(time(NULL));
    float intercept = 0;
    float slope1 = 0;
    float slope2 = 0;
    float slope3 = 0;
    float init_intercept = intercept;
    float init_slope1 = slope1;
    float init_slope2 = slope2;
    float init_slope3 = slope3;

    // Store the address of the x and y arrays into the pointers h_x and h_y (host_x and host_y)
    h_x1 = &x1[0];
    h_x2 = &x2[0];
    h_x3 = &x3[0];
    h_y = &y[0];
    h_intercept = &intercept;
    h_slope1 = &slope1;
    h_slope2 = &slope2;
    h_slope3 = &slope3;

    //Start measuring execution time of C tasks
    auto begin_cpu_run_time = std::chrono::high_resolution_clock::now();

    // EXECUTING CPU FUNCTION
    auto [intercept_cpu,slope1_cpu,slope2_cpu,slope3_cpu,number_of_iteration_cpu] = linear_regression_cpu(x1,x2,x3, y, init_intercept, init_slope1,init_slope2,init_slope3);

    auto end_cpu_run_time = std::chrono::high_resolution_clock::now();
    auto elapsed_cpu_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_run_time - begin_cpu_run_time);

    // std::cout << "CPU Results: intercept = " << intercept_cpu << " and slope: " << slope_cpu << " # Iterations: " << number_of_iteration_cpu << std::endl;

    std::cout << "---------------     CPU     ------------------"<<std::endl;
    std::cout << "GPU-implementation execution time [TOTAL] (micro s):: "<<elapsed_cpu_run_time.count() + elapsed_cpu_allocate.count()<<std::endl;
    std::cout << "CPU-implementation execution time (micro s): " << elapsed_cpu_run_time.count() << std::endl;
    std::cout << "CPU-allocate time (micro s): "<<elapsed_cpu_allocate.count()<<std::endl;

    //Save data on the savefile
    savefile << elapsed_cpu_run_time.count() << "\t" << elapsed_cpu_allocate.count() << std::endl;

    std::cout << "\nCPU Results:\n intercept = " << intercept_cpu << " slope 1: " << slope1_cpu << " slope 2: " << slope2_cpu <<  " slope 3: " << slope3_cpu << " # Iterations: " << number_of_iteration_cpu << std::endl;


//     _____ _____  _    _ 
//     / ____|  __ \| |  | |
//    | |  __| |__) | |  | |
//    | | |_ |  ___/| |  | |
//    | |__| | |    | |__| |
//     \_____|_|     \____/ 
                         
                        

    // Allocate memory on GPU for the device_x (d_x) and device_y (d_y) of earlier calculated size
    float* d_x1;
    float* d_x2;
    float* d_x3; 
    float* d_y; 
    float* d_intercept; 
    float* d_slope1; 
    float* d_slope2;
    float* d_slope3;
    float* d_results;

    cudaFree(0);

    auto begin_gpu_allocate = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_x1, input_size);
    cudaMalloc(&d_x2, input_size);
    cudaMalloc(&d_x3, input_size);
    cudaMalloc(&d_y, input_size);
    cudaMalloc(&d_results, error_size * numBlocks * sizeof(float));

    auto end_gpu_allocate = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu_allocate = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_allocate - begin_gpu_allocate);

    // Copy the values stored in pointer h_x and h_y into d_x and d_y
    // Transfer data from CPU memory to GPU memory.
    auto begin_gpu_copy = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_x1, h_x1, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, h_x3, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, input_size, cudaMemcpyHostToDevice);

    auto end_gpu_copy = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu_copy = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_copy - begin_gpu_copy);

    // std::cout<<"Copied\n";

    // Define stepsize for updating slope and intercept.
    // float learning_rate = LEARNING_RATE;

    //Start timing the procedure

    float j_error = std::numeric_limits<float>::max(); 

    int number_of_iteration_gpu = 0;

    auto total_gpu_allocate_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_kernel_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_copy_toHost_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_get_results_update_do = std::chrono::high_resolution_clock::duration::zero();
    auto total_gpu_free_do = std::chrono::high_resolution_clock::duration::zero();
    auto begin_gpu = std::chrono::high_resolution_clock::now();

    do{
        // Increase numeber of iterations
        number_of_iteration_gpu++;

        auto begin_gpu_allocate_do = std::chrono::high_resolution_clock::now();
        // ALlocate memory for the pointers to the intercept and slope
        cudaMalloc(&d_intercept, sizeof(float));
        cudaMalloc(&d_slope1, sizeof(float));
        cudaMalloc(&d_slope2, sizeof(float));
        cudaMalloc(&d_slope3, sizeof(float));
        auto end_gpu_allocate_do = std::chrono::high_resolution_clock::now();
        total_gpu_allocate_do += end_gpu_allocate_do - begin_gpu_allocate_do;


        // Copy the local value of the intercept and slope to the device memory.
        auto begin_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_intercept, h_intercept, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slope1, h_slope1, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slope2, h_slope2, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slope3, h_slope3, sizeof(float), cudaMemcpyHostToDevice);
        auto end_gpu_copy_toDevice_do = std::chrono::high_resolution_clock::now();
        total_gpu_copy_toDevice_do += end_gpu_copy_toDevice_do - begin_gpu_copy_toDevice_do;


        // Launch kernel on GPU with pointers to data in GPU memory

        auto begin_gpu_kernel = std::chrono::high_resolution_clock::now();

        simple_linear_regression<<<numBlocks,NUM_OF_THREADS>>>(d_x1,d_x2,d_x3, d_y, d_intercept, d_slope1,d_slope2,d_slope3, d_results, INPUT_SIZE);
        // Wait for all threads to return
        cudaDeviceSynchronize();

        auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
        total_gpu_kernel_do += end_gpu_kernel - begin_gpu_kernel;


        // Retrieve the GPU out value and store in host memory
        // auto begin_gpu_results_mem_copy = std::chrono::high_resolution_clock::now();
        auto begin_gpu_copy_toHost_do2 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_results, d_results, error_size * numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        auto end_gpu_copy_toHost_do2 = std::chrono::high_resolution_clock::now();
        total_gpu_copy_toHost_do += end_gpu_copy_toHost_do2 - begin_gpu_copy_toHost_do2;

        // Check if a CUDA error has occurred.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "--------------------- Error: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        auto begin_gpu_free_do = std::chrono::high_resolution_clock::now();

        // Free memory, on the next iteration we will allocate this memory again.
        cudaFree(d_intercept);
        cudaFree(d_slope1);
        cudaFree(d_slope2);
        cudaFree(d_slope3);

        auto end_gpu_free_do = std::chrono::high_resolution_clock::now();
        total_gpu_free_do += end_gpu_free_do - begin_gpu_free_do;

        float j_error = 0;
        float intercept_error = 0;
        float slope1_error = 0;
        float slope2_error = 0;
        float slope3_error = 0;

        // auto end_gpu_copy_toDevice_do2 = std::chrono::high_resolution_clock::now();
        // total_gpu_copy_toDevice_do = end_gpu_copy_toDevice_do2 - begin_gpu_copy_toDevice_do2;

        // printf("\n-----------------------------------------------------");
        // printf("\nSize of h_result : %lu",sizeof(h_results));

        
        auto begin_gpu_get_results_update_do = std::chrono::high_resolution_clock::now();

        for (int i=0; i<numBlocks*5; i++){
            // printf("\nh_result [%d] = %f",i,h_results[i]);
            if (i%ERROR_DIMENSIONS == 0){
                j_error += h_results[i];
            }
            if ((i-1)%ERROR_DIMENSIONS == 0){
                intercept_error += h_results[i];
            }
            if ((i-2)%ERROR_DIMENSIONS == 0){
                slope1_error += h_results[i];
            }
            if ((i-3)%ERROR_DIMENSIONS == 0){
                slope2_error += h_results[i];
            }
            if ((i-4)%ERROR_DIMENSIONS == 0){
                slope3_error += h_results[i];
            }
        }

        // Update intercept and slope based on errors
        float intercept_new = intercept - LEARNING_RATE * intercept_error;
        float slope1_new = slope1 - LEARNING_RATE * slope1_error;
        float slope2_new = slope2 - LEARNING_RATE * slope2_error;
        float slope3_new = slope3- LEARNING_RATE * slope3_error;

        // Update
        intercept = intercept_new;
        slope1 = slope1_new;
        slope2 = slope2_new;
        slope3 = slope3_new;
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

    auto begin_gpu_free = std::chrono::high_resolution_clock::now();

    // Free memory on GPU
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x3);
    cudaFree(d_y);

    auto end_gpu_free = std::chrono::high_resolution_clock::now();
    auto total_gpu_free = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_free - begin_gpu_free);

    //End timing and compute total execution time
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu);

    //Convert time counter in micro seconds
    auto total_gpu_allocate_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_allocate_do);
    auto total_gpu_copy_toDevice_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_copy_toDevice_do);
    auto total_gpu_kernel_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_kernel_do);
    auto total_gpu_copy_toHost_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_copy_toHost_do);
    auto total_gpu_get_results_update_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_get_results_update_do);
    auto total_gpu_free_do_micro = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_free_do);

    // Print out latest values for total error, and intercept and slope respective errors
    std::cout << "---------------     GPU     ------------------"<<std::endl;
    std::cout << "GPU-implementation execution time [TOTAL] (micro s): " <<elapsed_gpu_copy.count() + elapsed_gpu_allocate.count() + total_gpu_allocate_do_micro.count() + total_gpu_copy_toDevice_do_micro.count() + total_gpu_kernel_do_micro.count() + total_gpu_copy_toHost_do_micro.count() + total_gpu_get_results_update_do_micro.count()+total_gpu_free_do_micro.count()+total_gpu_free.count() <<std::endl;
    std::cout << "GPU-allocate time (micro s): "<<elapsed_gpu_allocate.count()<<std::endl;
    std::cout << "GPU-copy time (micro s): "<<elapsed_gpu_copy.count()<<std::endl;
    std::cout << "GPU-allocate do_cycle time (micro s): "<<total_gpu_allocate_do_micro.count()<<std::endl;
    std::cout << "GPU-copy do_cycle time (micro s): "<<total_gpu_copy_toDevice_do_micro.count()<<std::endl;
    std::cout << "GPU-kernel do_cycle(micro s): "<<total_gpu_kernel_do_micro.count()<<std::endl;
    std::cout << "GPU-copy do_cycle time (micro s): "<<total_gpu_copy_toHost_do_micro.count()<<std::endl;
    std::cout << "GPU-freeCuda do_cycle (micro s): "<<total_gpu_get_results_update_do_micro.count()<<std::endl;
    std::cout << "GPU-get results & update (micro s): "<<total_gpu_free_do_micro.count()<<std::endl;
    std::cout << "GPU-free CUDA (micro s): "<<total_gpu_free.count()<<std::endl;

    //save data on the savefile
    // savefile<<elapsed_gpu.count()<<"\t"<<elapsed_gpu_allocate.count()<<"\t"<<elapsed_gpu_copy.count()<<"\t"<<total_gpu_allocate_do_micro.count()<<"\t"<<total_gpu_copy_toDevice_do_micro.count()<<"\t"<<total_gpu_kernel_do_micro.count()<<"\t"<<total_gpu_get_results_update_do_micro.count()<<std::endl;
    // savefile<<elapsed_gpu.count()<<"\t"<<elapsed_gpu_allocate.count()<<"\t"<<elapsed_gpu_copy.count()<<"\t"<<total_gpu_allocate_do_micro.count()<<"\t"<<total_gpu_copy_toDevice_do_micro.count()<<"\t"<<total_gpu_kernel_do_micro.count()<<"\t"<<total_gpu_copy_toHost_do_micro.count()<<"\t"<<total_gpu_get_results_update_do_micro.count()<<std::endl;
    savefile<<elapsed_gpu_allocate.count()<<"\t"<<elapsed_gpu_copy.count()<<"\t"<<total_gpu_allocate_do_micro.count()<<"\t"<<total_gpu_copy_toDevice_do_micro.count()<<"\t"<<total_gpu_kernel_do_micro.count()<<"\t"<<total_gpu_copy_toHost_do_micro.count()<<"\t"<<total_gpu_get_results_update_do_micro.count()<<"\t"<<total_gpu_free.count()<<std::endl;

    std::cout << "GPU Results:\n intercept = " << intercept << " slope1: " << slope1 << " slope2: " << slope2 << " slope3: " << slope3 << " # Iterations: "<< number_of_iteration_gpu <<  std::endl;

}
    return 0;
}


