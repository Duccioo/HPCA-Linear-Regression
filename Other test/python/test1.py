import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate random dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20)

# Add a column of ones to X to account for the intercept term
X = np.column_stack((np.ones(len(X)), X))

# Compute normal equation
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Plot initial data points
plt.scatter(X[:,1], y, alpha=0.5)

# Plot regression line
x_vals = np.array([X[:,1].min(), X[:,1].max()])
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, '--', color='red')

# Add labels and title to plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')

# Show plot
plt.show()