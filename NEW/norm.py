import pandas as pd

def normalize(file_name):
    # leggi il file
    df = pd.read_csv(file_name, header=None)
    
    # normalizza le colonne dividendo per il massimo
    normalized_df = (df-df.min())/(df.max()-df.min())

    normalized_df.to_csv(file_name, index=False)
