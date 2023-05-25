import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import os

# Define the list of new column names
new_column_names = [
    'Allocate',
    'Copy',
    'Allocate Do Cycle',
    'Copy To Device',
    'Execution',
    'Copy To Host',
    'FreeCuda Do Cycle',
    'Get Results & Update',
    'FreeCuda'
]

font = {'weight': 'bold',
        'size': 12}

matplotlib.rc('font', **font)

folder_path = os.path.join('..','result')  # Specify the folder path

# Sort the file names alphabetically
file_names = os.listdir(folder_path)
file_names.sort(key=lambda x: [int(c) if c.isdigit(
) else c.lower() for c in re.split(r'(\d+)', x)], reverse=True)


all_dataframe = pd.DataFrame()

for filename in file_names:
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        print(file_path)
        # Perform desired operations on each file
        # print(file_path)

        params = file_path.split('_')
        # print(params[1])

        # PARAMETERS
        # 1 : number of repetitions
        # 2 : number of threads
        # 3 : max j error
        # 4 : learning rate

        # Read the file into a list of lines
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Process the lines and create a list of dictionaries
        data_cpu = []
        data_gpu = []
        for line in lines:
            values = line.strip().split('\t')
            if len(values) == 2:
                data_cpu.append(
                    {'Execution': int(values[0]), 'Allocate': int(values[1])})
            elif len(values) > 2:
                data_gpu.append({'Value{}'.format(i+1): int(value)
                                for i, value in enumerate(values)})

        # Create a DataFrame from the list of dictionaries
        data_pd_cpu = pd.DataFrame(data_cpu)
        data_pd_gpu = pd.DataFrame(data_gpu)

        # Create a dictionary with the new column names
        new_columns = {old_col: new_col for old_col,
                       new_col in zip(data_pd_gpu.columns, new_column_names)}

        # Rename the columns in the DataFrame
        data_pd_gpu.rename(columns=new_columns, inplace=True)

        cpu_mean = data_pd_cpu.mean()
        # print(cpu_mean)

        gpu_mean = data_pd_gpu.mean()
        # print(gpu_mean)

        # Convert Series to DataFrames
        cpu_mean = cpu_mean.to_frame().transpose()
        gpu_mean = gpu_mean.to_frame().transpose()

        # print(cpu_mean)
        # print(gpu_mean)

        gpu_name = 'GPU ['+params[5]+' machine]'

        cpu_name = 'CPU ['+params[5]+' machine]'

        gpu_mean.index = [gpu_name]
        cpu_mean.index = [cpu_name]

        gpu_mean = pd.concat([gpu_mean, cpu_mean])

        all_dataframe = pd.concat([all_dataframe, gpu_mean])

all_dataframe.plot.barh(stacked=True)
plt.xlabel('Microseconds [μs]')
plt.grid(axis='x', linestyle='--')
plt.show()
