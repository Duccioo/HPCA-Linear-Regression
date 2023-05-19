import pandas as pd

# leggi il file
df = pd.read_csv('Other test/single-linear-regression/data/output2.csv', header=None)

# normalizza le colonne dividendo per il massimo
df = df.apply(lambda x: x / x.max(), axis=0)

df.to_csv('Other test/single-linear-regression/data/output_norm.csv', index=False, header=False)