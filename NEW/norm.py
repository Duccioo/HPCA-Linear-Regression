import pandas as pd


def normalize(file_name):
    print(f"file_name : {file_name}")

    # Read the file
    df = pd.read_csv(file_name)

    # Exclude the first row (header)
    df = df.iloc[1:]

    # Convert columns to numeric data type
    df = df.apply(pd.to_numeric, errors='coerce')

    numeric_columns = df.select_dtypes(include=['float64']).columns

    # Normalize numeric columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / \
        (df[numeric_columns].max() - df[numeric_columns].min())

    # Save normalized dataset to a new file
    df.to_csv(file_name, index=False, header=True)

    print("Dataset Normalized")
