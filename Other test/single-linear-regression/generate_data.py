import numpy as np
import matplotlib.pyplot as plt

def generate_data_and_save(n_samples, slope, intercept, noise_std, filename):
    """
    Generates data from a linear equation with added noise and saves it to a file.

    Args:
        n_samples (int): Number of data points to generate.
        slope (float): Slope of the linear equation.
        intercept (float): Intercept of the linear equation.
        noise_std (float): Standard deviation of the Gaussian noise to add to the data.
        filename (str): Name of the file to save the data to.
    """
    # Generate x values
    x = np.linspace(0, 1, n_samples)

    # Generate y values from linear equation with added noise
    y = slope * x + intercept + np.random.normal(0, noise_std, n_samples)
    
    plt.scatter(x,y)
    plt.title(filename)
    plt.show()

    # Save data to file
    np.savetxt(filename, np.column_stack((x, y)), delimiter=',', header='x,y', comments='')

    print(f"Data saved to {filename}")
    
n_samples = 10
generate_data_and_save(n_samples=n_samples, slope=2, intercept=1, noise_std=0.5, filename="Other test/single-linear-regression/data/genereted/data_"+str(n_samples)+"_2_1.csv")
