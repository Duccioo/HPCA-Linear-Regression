import pandas as pd


def get_result(x1, x2, x3):
    intercept = 0.27301457524299622
    slope1 = 0.19559884071350098
    slope2 = 0.25461623072624207
    slope3 = 0.12938199937343597
    return intercept + slope1 * x1 + slope2 * x2 + slope3 * x3


test_data = '..data/mock_dataset/test.csv'  # Specify the folder path

num_data = 16000

test_data_df = pd.read_csv(test_data)

print("max of y: ", test_data_df['y'].max())
print("min of y: ", test_data_df['y'].min())

error = 0

for _, row in test_data_df.iterrows():
    predict = get_result(row['x1'], row['x2'], row['x3'])
    error += abs(predict - row['y'])

print(error / num_data)
print("Accuracy: ", 100 - (error / num_data * 100))
