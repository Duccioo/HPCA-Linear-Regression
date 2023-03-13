import numpy as np
import matplotlib.pyplot as plt
import time

# Choose the number of data to generate for each variable
num_data = 20000000

# Generate random dataset with two independent variables
np.random.seed(42)
X1 = np.random.rand(num_data, 1) * 30
X2 = np.random.rand(num_data, 1) * 50
X3 = np.random.rand(num_data, 1) * 70
X4 = np.random.rand(num_data, 1) * 7
X5 = np.random.rand(num_data, 1) * 12
X6 = np.random.rand(num_data, 1) * 123
X = np.c_[X1, X2]
y = 7 * X1 + 5 * X2 - 7 * X3 + 12 * X4 - 2 * X5 +0.5 * X6 + np.random.randn(num_data, 1) 

# Add a column of ones to X for bias term
X_b = np.c_[np.ones((num_data, 1)), X]

# Compute normal equation to find optimal parameters
start_time = time.time()
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
end_time = time.time()

print("CPU TIME: ",end_time - start_time," sec")

# # Print optimal parameters
# print('Optimal parameters:')
# print(theta_best)

# # Predict values for new input
# X_new = np.array([[1, 5, 2], [1, 3, 4]])
# y_predict = X_new.dot(theta_best)

# # Print predicted values for new input
# print('Predicted values for new input:')
# print(y_predict)

# # Plot the dataset and regression plane
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1, X2, y, c='b', marker='.')
# X1_new, X2_new = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 5, 50))
# X_new = np.c_[np.ones((X1_new.size, 1)), X1_new.ravel(), X2_new.ravel()]
# y_predict = X_new.dot(theta_best).reshape(X1_new.shape)
# ax.plot_surface(X1_new, X2_new, y_predict, alpha=0.5)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('y')
# plt.show()
