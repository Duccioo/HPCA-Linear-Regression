import pandas as pd

# leggi il file
df = pd.read_csv(
    'Other test/single-linear-regression/data/output2.csv', header=None)

# normalizza le colonne dividendo per il massimo
# df = df.apply(lambda x: x / x.max(), axis=0)
normalized_df = (df-df.min())/(df.max()-df.min())

normalized_df.to_csv('NEW/data/output_norm22.csv',
                     index=False)
