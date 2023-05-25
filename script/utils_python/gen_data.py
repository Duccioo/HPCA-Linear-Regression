import csv
import random
from norm import normalize


def generate_dataset(n, intercept, slope1, slope2, slope3):
    dataset = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(0, 10)
        noise = random.uniform(-1, 1)
        y = intercept + slope1 * x1 + slope2 * x2 + slope3 * x3 + noise
        dataset.append([x1, x2, x3, y])

    return dataset


def save_dataset_to_csv(dataset, filename):
    print(filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x1', 'x2', 'x3', 'y'])
        writer.writerows(dataset)


# Parameters
n = 500000  # Number of samples
intercept = 2.5
slope1 = 1.2
slope2 = 0.8
slope3 = -0.5

# Generate dataset
dataset = generate_dataset(n, intercept, slope1, slope2, slope3)

path = 'NEW/data/genereted/4D/'+str(n)+'_dataset.csv'

# Save dataset to CSV
save_dataset_to_csv(dataset, path)

# Normalize dataset
dataset = normalize(path)
