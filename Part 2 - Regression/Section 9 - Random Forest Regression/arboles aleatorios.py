# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:04:22 2019

@author: darub
"""
# regresion con arboles aleatorios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values


# Dividir entrenamiento y testeo
# son pocos datos no se usa

# crear un objeto de arboles aleatorios
from sklearn.ensemble import RandomForestRegressor
# n_estimators es el numero de arboles, random_state semilla 0
randon_forest = RandomForestRegressor(n_estimators=300, random_state=0)
randon_forest.fit(x,y)

y_pred = randon_forest.predict([[6.5]])



# Graficar arbol
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y, color="red")
plt.plot(x_grid, randon_forest.predict(x_grid), color="blue")
plt.title("Arboles aleatorios")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()



























