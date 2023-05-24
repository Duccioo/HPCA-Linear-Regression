import pandas as pd


def get_result(x1, x2, x3):
    intercept = 0.14544570446014404
    slope1 = 0.028964448720216751
    slope2 = 0.83468538522720337
    slope3 = 0.00063588534248992801
    return intercept + slope1 * x1 + slope2 * x2 + slope3 * x3


test_data = 'NEW/data/test_2.csv'  # Specify the folder path

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
