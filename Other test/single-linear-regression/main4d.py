import numpy as np
import csv

INPUT_SIZE = 5
ERROR_DIMENSIONS = 5
MAX_J_ERROR = 0.025
LEARNING_RATE = 0.001
MAX_ITER = 50000


def load_data(filename):
    x1 = np.zeros(INPUT_SIZE)
    x2 = np.zeros(INPUT_SIZE)
    x3 = np.zeros(INPUT_SIZE)
    y = np.zeros(INPUT_SIZE)

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for i, row in enumerate(reader):
            x1[i], x2[i], x3[i], y[i] = map(float, row)

    return x1, x2, x3, y


def linear_regression_cpu(x1, x2, x3, y, intercept1, slope1, slope2, slope3):
    errors = np.zeros(ERROR_DIMENSIONS)
    errors[0] = 500000
    iter = 0

    while errors[0] > MAX_J_ERROR and iter < MAX_ITER:
        iter += 1
        for i in range(INPUT_SIZE):
            y_pred = intercept1 + slope1 * x1[i] + slope2 * x2[i] + slope3 * x3[i]
            errors[0] += 0.5 * ((y[i] - y_pred) ** 2)
            errors[1] += -(y[i] - y_pred)
            errors[2] += -(y[i] - y_pred) * x1[i]
            errors[3] += -(y[i] - y_pred) * x2[i]
            errors[4] += -(y[i] - y_pred) * x3[i]

        errors[0] /= INPUT_SIZE
        intercept1_new = intercept1 - LEARNING_RATE * errors[1]
        slope1_new = slope1 - LEARNING_RATE * errors[2]
        slope2_new = slope2 - LEARNING_RATE * errors[3]
        slope3_new = slope3 - LEARNING_RATE * errors[4]
        print(iter, errors[0])

        intercept1 = intercept1_new
        slope1 = slope1_new
        slope2 = slope2_new
        slope3 = slope3_new

    return intercept1_new, slope1_new, slope2_new, slope3_new, errors[0]


def main():
    x1, x2, x3, y = load_data("data/dati.csv")
    print("Data Loaded.")

    intercept1 = 0.1
    intercept2 = 0.2
    slope1 = 0.3
    slope2 = 0.4

    results = linear_regression_cpu(
        x1, x2, x3, y, intercept1, intercept2, slope1, slope2
    )
    intercept1_new, slope1_new, slope2_new, slope3_new, error = results

    print("ERROR: =", error)
    print("intercept1 =", intercept1_new)
    print("slope1 =", slope1_new)
    print("slope2 =", slope2_new)
    print("slope3 =", slope3_new)


if __name__ == "__main__":
    main()
