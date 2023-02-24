import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm


# Caricamento dei dati dal file CSV
data = pd.read_csv("insurance.csv")

# Encoding delle colonne con valori categorici
data = pd.get_dummies(data, columns=["sex", "smoker", "region"])

# Preparazione dei dati per la regressione lineare
X = data.drop("charges", axis=1)
y = data["charges"]


# Creazione del modello di regressione lineare e addestramento
model = LinearRegression()
model.fit(X, y)

# SVM = svm.SVR()
# SVM.fit(X, y)

# Stampa dei parametri della regressione lineare
print("Intercept:", model.intercept_)
print("Slope for age:", model.coef_[0])
print("Slope for sex (female):", model.coef_[1])
print("Slope for smoker (yes):", model.coef_[2])
print("Slope for region (northeast):", model.coef_[3])

# Calcolo del costo (errore quadratico medio)
mse = ((model.predict(X) - y) ** 2).mean()
print("MSE Linear:", mse)
accuracy = round(r2_score(model.predict(X), y), 2) * 100
print("Accuracy Linear:", accuracy)


# Grafico dei dati e della retta di regressione
plt.scatter(X["age"], y, color="black")
plt.plot(X["age"], model.predict(X), color="blue", linewidth=3)
plt.xlabel("age")
plt.ylabel("charges")
plt.legend(["Regression line", "Data points"])
plt.show()
