import numpy as np
import matplotlib.pyplot as plt
import time

# Generate random dataset
np.random.seed(0)
n_samples = 20
X = np.linspace(0, 10, n_samples)
y = 50 + 0.5 * X + np.random.normal(0, 200, n_samples)

# Add a column of ones to X to account for the intercept term
X_ones = np.hstack([np.ones((n_samples, 1)), X.reshape(-1, 1)])

# Compute normal equation
start_time = time.time()
XtX = np.dot(X_ones.T, X_ones)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(X_ones.T, y)
theta = np.dot(XtX_inv, Xty)
end_time = time.time()

# Print CPU run time
cpu_time = end_time - start_time
print(f"CPU run time: {cpu_time:.5f} seconds")

# Plot initial data points
plt.scatter(X, y, s=5)

# Plot regression line
x_vals = np.array([X.min(), X.max()])
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, "--r")

# Add labels and title to plot
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")

# Show plot
plt.show()

# Compute error on predicted values
y_pred = np.dot(X_ones, theta)
mse = np.mean((y_pred - y)**2)
print(f"Mean Squared Error: {mse:.2f}")
