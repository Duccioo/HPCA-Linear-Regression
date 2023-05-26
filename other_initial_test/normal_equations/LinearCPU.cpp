#include <Eigen/Dense>
#include "matplotlibcpp.h"
#include <random>
#include <chrono>

namespace plt = matplotlibcpp;

int main() {
  // Generate random dataset
  std::default_random_engine generator{0};
  const int n_samples = 20000000;
  Eigen::VectorXd X(n_samples);
  Eigen::VectorXd y(n_samples);

  auto start_time_gen = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_samples; ++i) {
    X(i) = i * 0.5;
    y(i) = 50 + 0.5 * X(i) + std::normal_distribution<double>(0, 200)(generator);
  }
  auto end_time_gen = std::chrono::high_resolution_clock::now();
     // Print CPU run time
  std::chrono::duration<double> gen_time = end_time_gen - start_time_gen;
  std::cout << "Generating time: " << gen_time.count() << " seconds" << std::endl;

  // Add a column of ones to X to account for the intercept term
  Eigen::MatrixXd X_ones(n_samples, 2);
  X_ones.col(0).setOnes();
  X_ones.col(1) = X;

  // Compute normal equation
  auto start_time = std::chrono::high_resolution_clock::now();
  const auto XtX = X_ones.transpose() * X_ones;
  const auto XtX_inv = XtX.inverse();
  const auto Xty = X_ones.transpose() * y;
  const auto theta = XtX_inv * Xty;
  auto end_time = std::chrono::high_resolution_clock::now();

   // Print CPU run time
  std::chrono::duration<double> cpu_time = end_time - start_time;
  std::cout << "CPU run time: " << cpu_time.count() << " seconds" << std::endl;

//   // Plot initial data points
//   std::vector<double> x_vec(X.data(), X.data() + X.size());
//   std::vector<double> y_vec(y.data(), y.data() + y.size());
// //   plt::scatter(x_vec, y_vec, 5);

//   // Plot regression line
//   const double x_min = X.minCoeff();
//   const double x_max = X.maxCoeff();
//   std::vector<double> x_vals{x_min, x_max};
//   std::vector<double> y_vals{theta(0) + theta(1) * x_min, theta(0) + theta(1) * x_max};

//   plt::plot(x_vals, y_vals, "--r");

//   // Add labels and title to plot
//   plt::xlabel("X");
//   plt::ylabel("y");
//   plt::title("Linear Regression");

//   // Show plot
//   plt::show();

  // Compute error on predicted values
  auto start_time_mse = std::chrono::high_resolution_clock::now();
  const auto y_pred = X_ones * theta;
  const auto mse = (y_pred - y).squaredNorm() / n_samples;
  auto end_time_mse = std::chrono::high_resolution_clock::now();

//   std::printf("Mean Squared Error: %.2f\n", mse);

     // Print CPU run time
  std::chrono::duration<double> mse_time = end_time_mse - start_time_mse;
  std::cout << "MSE run time: " << mse_time.count() << " seconds" << std::endl;

  return 0;
}
