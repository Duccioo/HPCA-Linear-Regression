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

// #include "linear_regression.cuh"

#define INPUT_SIZE 5
#define ERROR_DIMENSIONS 5
// #define NUM_OF_THREADS 256 
#define MAX_J_ERROR 2
#define LEARNING_RATE 0.00001 
#define MAX_ITER 50000
// #define NUM_REP 10

// Load 4D data
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

// Linear regression function 
std::tuple<float,float,float,float,float> linear_regression_cpu(const std::array<float, INPUT_SIZE> &x1, 
                                                              const std::array<float, INPUT_SIZE> &x2,
                                                              const std::array<float, INPUT_SIZE> &x3, 
                                                              const std::array<float, INPUT_SIZE> &y, 
                                                              float intercept1, float slope1,  
                                                              float slope2, float slope3) {
    // Calculate errors
    float errors[ERROR_DIMENSIONS] = {0}; 
    errors[0] = 1000000;
    float intercept1_new = 0;
    float slope1_new     = 0;
    float slope2_new     = 0;
    float slope3_new     = 0;
    int iter = 0;
    while(errors[0] > MAX_J_ERROR){
        iter++;
        if (iter > MAX_ITER){
            break;
        }
        for (int i = 0; i < INPUT_SIZE; ++i) {
            // Predict output based on current coefficients
            float y_pred = intercept1 + slope1*x1[i] + slope2*x2[i] + slope3*x3[i];
            // Calculate total error
            errors[0] += 0.5f * pow((y[i] - y_pred), 2);
            // Calculate error for each coefficient
            errors[1] += -(y[i] - y_pred);
            errors[2] += -(y[i] - y_pred)*x1[i];
            errors[3] += -(y[i] - y_pred)*x2[i];
            errors[4] += -(y[i] - y_pred)*x3[i]; 
        }
        // Normalize errors
        errors[0] = errors[0] / INPUT_SIZE;
        
        // Update coefficients based on errors
        intercept1_new = intercept1 - LEARNING_RATE * errors[1]; 
        intercept1 = intercept1_new;
        slope1_new     = slope1 - LEARNING_RATE * errors[2]; 
        slope1 = slope1_new;
        slope2_new     = slope2 - LEARNING_RATE * errors[3];
        slope2=slope2_new;
        slope3_new     = slope3 - LEARNING_RATE * errors[4];
        slope3=slope3_new;
        std::cout<<iter<<" "<<errors[0]<<"\n";
    }
    
    // Return updated coefficients and number of iterations
    return std::tuple<float, float, float, float, float>{intercept1_new, slope1_new, slope2_new, slope3_new, errors[0]};
}

int main(int argc, char **argv) {
    // Load 4D data
    std::array<float, INPUT_SIZE> x1; 
    std::array<float, INPUT_SIZE> x2;
    std::array<float, INPUT_SIZE> x3;
    std::array<float, INPUT_SIZE> y;
    load_data("data/dati.csv", x1, x2, x3, y);
    std::cout << "Data Loaded."<< std::endl;;

    // Initial coefficients
    float intercept1 = 0.1; 
    float intercept2 = 0.2;
    float slope1 = 0.3;
    float slope2 = 0.4;

    std::tuple<float, float, float, float, float> results = linear_regression_cpu(x1, x2, x3, y,  intercept1, intercept2, slope1, slope2);
    float intercept1_new = std::get<0>(results);
    //
    std::cout << "ERROR: = " << std::get<4>(results) << std::endl;
    // Print results
    std::cout << "intercept1 = " << std::get<0>(results) << std::endl;
    std::cout << "slope1 = " << std::get<1>(results) << std::endl;
    std::cout << "slope2 = " << std::get<2>(results) << std::endl; 
    std::cout << "slope3 = " << std::get<3>(results) << std::endl;
}