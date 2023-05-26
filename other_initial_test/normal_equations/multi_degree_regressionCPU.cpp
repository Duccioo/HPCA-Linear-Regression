#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

// Define the degree of the polynomial
const int degree = 1;

// Define the number of samples to generate
const int num_samples = 15;

int main() {
    // Generate a dataset with a polynomial relationship
    srand(0);
    vector<double> x(num_samples);
    vector<double> y(num_samples);
    double step = 20.0 / (num_samples - 1);
    for (int i = 0; i < num_samples; i++) {
        x[i] = -10 + i * step;
        y[i] = 0;
        for (int j = 0; j <= degree; j++) {
            y[i] += pow(x[i], j);
        }
        y[i] += 100 * (double)rand() / RAND_MAX ;
    }

    // Fit a polynomial regression model to the data
    auto start_time = chrono::high_resolution_clock::now();
    vector<vector<double>> X(num_samples, vector<double>(degree + 1));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j <= degree; j++) {
            X[i][j] = pow(x[i], j);
        }
    }
    vector<double> coeffs(degree + 1);
    for (int i = 0; i <= degree; i++) {
        double sum = 0;
        for (int j = 0; j < num_samples; j++) {
            sum += X[j][i] * y[j];
        }
        coeffs[i] = sum;
    }
    for (int i = degree; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double factor = X[j][i] / X[i][i];
            for (int k = i; k >= 0; k--) {
                X[j][k] -= factor * X[i][k];
            }
            y[j] -= factor * y[i];
        }
    }
    for (int i = 0; i <= degree; i++) {
        coeffs[i] = y[i] / X[i][i];
    }
    auto end_time = chrono::high_resolution_clock::now();

    // Print the CPU execution time
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "CPU execution time: " << duration.count() / 1000.0 << " seconds" << endl;

    // Plot the data and the regression line
    vector<double> x_fit(100);
    step = 20.0 / 99;
    for (int i = 0; i < 100; i++) {
        x_fit[i] = -10 + i * step;
    }
    vector<double> y_fit(100);
    for (int i = 0; i < 100; i++) {
        y_fit[i] = 0;
        for (int j = 0; j <= degree; j++) {
            y_fit[i] += coeffs[j] * pow(x_fit[i], j);
        }
    }
    plt::scatter(x, y);
    plt::plot(x_fit, y_fit, "r");
    plt::show();

    return 0;
}
