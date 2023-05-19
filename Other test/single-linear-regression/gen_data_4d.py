import numpy as np

# Parametri della distribuzione
intercept1 = 2.0
slope1 = 0.5
slope2 = 1.0
slope3 = -0.5

# Numero di dati da generare
num_data = 1000

# Dati di input fissati
x1 = np.linspace(1.0, 5.0, num_data)
x2 = np.linspace(0.5, 4.5, num_data)
x3 = np.linspace(0.1, 0.5, num_data)

# Generazione dei dati y_pred
y_pred = intercept1 + slope1 * x1 + slope2 * x2 + slope3 * x3

# Creazione di un array multidimensionale con i dati
data = np.column_stack((x1, x2, x3, y_pred))

# Salvataggio dei dati in un file CSV
np.savetxt("Other test/single-linear-regression/data/dati.csv", data, delimiter=",", header="x1,x2,x3,y_pred", comments="")

print("Dati salvati correttamente nel file 'dati.csv'.")
