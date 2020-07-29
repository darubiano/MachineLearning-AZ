# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:34:58 2019

@author: darub
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
# tomar todos los valores menos la ultima experiencia
x = dataset.iloc[:,:-1].values
# tomar los solarios
y = dataset.iloc[:,1].values

# dividir datos para entrenamiento y testeo, tomar 10 de 30 elementos para testear
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
##############################################################################
# crear modelo de regresion lineal simple con el conjunto de entrenamiento
regresion = LinearRegression()
regresion.fit(x_train,y_train)

# Predecir el conjunto de testeo, solo toma los valores x para predecir los valores de y_test
y_pred = regresion.predict(x_test)

# Visulaizar los resultados de entranamiento
plt.scatter(x_train, y_train, color= "red")
plt.plot(x_train, regresion.predict(x_train), color ="blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visulaizar los resultados de test, la recta es la misma 
plt.scatter(x_test, y_test, color= "red")
plt.plot(x_train, regresion.predict(x_train), color ="blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# resumen de los datos.
summary = dataset.describe()
print(summary)


