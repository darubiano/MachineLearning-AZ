# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:56:31 2019

@author: darub
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# Entrenar el modelo, random_state generar semilla con valor 0
reg_tree = DecisionTreeRegressor(random_state= 0)
reg_tree.fit(x,y)

# Prediccion
y_pred = reg_tree.predict([[6.5]])


# Graficar modelo, divide los datos en segmentos si es mayor que 10 siempre dara el valor del salario en 10
x_grid = np.arange(min(x), max(x), 0.1) 
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color="red")
plt.plot(x_grid , reg_tree.predict(x_grid), color="blue")
plt.title("Modelo arboles de decision por regresion")
plt.xlabel("Posicion del empleado años")
plt.ylabel("Sueldo (en $)")
plt.show()














