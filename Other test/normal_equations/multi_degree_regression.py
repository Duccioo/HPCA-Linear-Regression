import numpy as np
import matplotlib.pyplot as plt
import time

# Define the degree of the polynomial
degree = 1

# Define the number of samples to generate
num_samples = 1000000

# Generate a dataset with a polynomial relationship
np.random.seed(0)
x = np.linspace(-10, 10, num=num_samples)
y = np.sum([x**i for i in range(degree + 1)], axis=0) + 100*np.random.randn(num_samples)

# Fit a polynomial regression model to the data
start_time = time.time()
X = np.vstack([x**i for i in range(degree + 1)]).T
coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
end_time = time.time()

# Print the CPU execution time
print("CPU execution time:", end_time - start_time, "seconds")

# Plot the data and the regression line
plt.scatter(x, y, label='data')
x_fit = np.linspace(-10, 10, num=100)
y_fit = np.sum([coeffs[i]*x_fit**i for i in range(degree + 1)], axis=0)
plt.plot(x_fit, y_fit, 'r', label='polynomial regression')
plt.legend()
plt.show()
