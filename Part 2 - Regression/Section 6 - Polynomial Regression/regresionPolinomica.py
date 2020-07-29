# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:56:25 2019

@author: darub
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# funciona para hacer entrenamiento lineal multilineal y polinomial LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
dataset = pd.read_csv("Position_Salaries.csv")

#Tomar los valores desde la columna 1 y sin la ultima
x = dataset.iloc[:,1:-1].values
# Tomar la ultima columna
y = dataset.iloc[:,-1].values

# no se divide en trenamiento y testeo por que se son pocos datos y se buscan los valores intermedios

# Ajustar la regresion lienal con el dataset
lin_reg = LinearRegression()
lin_reg.fit(x,y)
# predecir el valor de 11 cuanto deberian pagar
y_pred_lin = lin_reg.predict([[11]])

# Ajustar la regresion polinomica con el dataset, valor por defecto alcuadrado degree
poly_reg = PolynomialFeatures(degree=3)
# elevar los datos al cuadrado
x_poly = poly_reg.fit_transform(x)

#Entrenar el modelo con los datos elevandos al cuadrado
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
y_pred_lin_2 = lin_reg_2.predict([[1,1,11,121]])


# Visualizacion de los resultados del modelo lienal
plt.scatter(x,y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Modelo de regresin lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizacion de los resultados del modelo polinomico
# crear un rango de valores de 0.1 en 0.1 de x
x_grid = np.arange(min(x), max(x), 0.1)
# redimencionar para que el vector sea una columna
x_grid =  x_grid.reshape(len(x_grid),1)

plt.scatter(x,y, color="red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Modelo de regresin polinomial")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()



























