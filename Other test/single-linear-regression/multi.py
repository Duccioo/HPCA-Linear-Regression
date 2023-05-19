import numpy as np
import matplotlib.pyplot as plt


# Generate random data for multilinear regression
def generate_data(num_samples, num_features):
    np.random.seed(0)
    X = np.random.rand(num_samples, num_features)
    true_coeffs = np.random.rand(num_features)
    noise = np.random.normal(0, 0.1, num_samples)
    y = np.dot(X, true_coeffs) + noise
    return X, y, true_coeffs


# Perform multilinear regression using gradient descent
def multilinear_regression(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    coeffs = np.zeros(num_features)

    for _ in range(num_iterations):
        y_pred = np.dot(X, coeffs)
        error = y_pred - y
        gradient = np.dot(X.T, error) / num_samples
        coeffs -= learning_rate * gradient

    return coeffs


# Generate random data
num_samples = 100
num_features = 4
X, y, true_coeffs = generate_data(num_samples, num_features)

# Perform multilinear regression
learning_rate = 0.0001
num_iterations = 100000
coeffs = multilinear_regression(X, y, learning_rate, num_iterations)

# Print true coefficients and estimated coefficients
print("True Coefficients:", true_coeffs)
print("Estimated Coefficients:", coeffs)

# Calculate the root mean squared error (RMSE)
mse = np.mean((y - np.dot(X, coeffs)) ** 2)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot the true and predicted values
y_pred = np.dot(X, coeffs)
plt.scatter(range(num_samples), y, color="blue", label="True")
plt.scatter(range(num_samples), y_pred, color="red", label="Predicted")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.legend()
plt.savefig("fig.png")
