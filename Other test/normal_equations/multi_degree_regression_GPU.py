import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time

# Define the degree of the polynomial
degree = 1

# Define the number of samples to generate
num_samples = 15

# Generate a dataset with a polynomial relationship
np.random.seed(0)
x = np.linspace(-10, 10, num=num_samples)
y = np.sum([x**i for i in range(degree + 1)], axis=0) + 100*np.random.randn(num_samples)

# Fit a polynomial regression model to the data
start_time = time.time()

@cuda.jit
def poly_regression(x, y, coeffs):
    n = len(x)
    X = cuda.local_array((n, degree + 1), dtype=np.float32)
    for i in range(n):
        for j in range(degree + 1):
            X[i][j] = x[i]**j
    X_T = cuda.local_array((degree + 1, n), dtype=np.float32)
    for i in range(degree + 1):
        for j in range(n):
            X_T[i][j] = X[j][i]
    XTX = cuda.local_array((degree + 1, degree + 1), dtype=np.float32)
    for i in range(degree + 1):
        for j in range(degree + 1):
            XTX[i][j] = 0
            for k in range(n):
                XTX[i][j] += X_T[i][k] * X[k][j]
    XTY = cuda.local_array((degree + 1,), dtype=np.float32)
    for i in range(degree + 1):
        XTY[i] = 0
        for j in range(n):
            XTY[i] += X_T[i][j] * y[j]
    coeffs[0] = (XTX[1][1] * XTY[0] - XTX[0][1] * XTY[1]) / (XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0])
    coeffs[1] = (XTX[0][0] * XTY[1] - XTX[1][0] * XTY[0]) / (XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0])

coeffs = np.zeros((degree + 1,), dtype=np.float32)
poly_regression[num_samples, 1](x, y, coeffs)

end_time = time.time()

# Print the GPU execution time
print("GPU execution time:", end_time - start_time, "seconds")

# Plot the data and the regression line
plt.scatter(x, y, label='data')
x_fit = np.linspace(-10, 10, num=100)
y_fit = np.sum([coeffs[i]*x_fit**i for i in range(degree + 1)], axis=0)
plt.plot(x_fit, y_fit, 'r', label='polynomial regression')
plt.legend()
plt.show()
