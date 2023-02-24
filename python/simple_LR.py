import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Numero di dati da generare
N = 500

# Generazione casuale dei dati
x1 = np.random.normal(0, 1, size=N)

intercept = np.random.normal(0, 0.5, size=N)
noise = np.random.normal(0, 0.5, size=N)
y = 3 * x1 + intercept + noise

# Creazione di una matrice X contenente le variabili indipendenti
X = x1


# Creazione del modello di regressione lineare e addestramento
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)


# Stampa dei parametri della regressione lineare
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
# Calcolo del costo (errore quadratico medio)
mse = ((model.predict(X.reshape(-1, 1)) - y) ** 2).mean()
print("MSE:", mse)
accuracy = round(r2_score(model.predict(X.reshape(-1, 1)), y), 2) * 100
print("Accuracy:", accuracy)

# Grafico dei dati e della retta di regressione
fig, ax = plt.subplots()
ax.scatter(x1, y, color="red", alpha=0.5, label="Data")
plt.plot(X, model.predict(X.reshape(-1, 1)), color="blue", linewidth=3)
plt.xlabel("age")
plt.ylabel("charges")
plt.legend(["Regression line", "Data points"])
plt.show()
