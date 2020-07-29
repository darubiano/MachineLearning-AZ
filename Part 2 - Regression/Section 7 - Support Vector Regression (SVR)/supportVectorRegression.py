# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:29:29 2019

@author: darub
"""
#SVR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# no se divide el dataset para entrenamiento son pocos datos

# Escalado de variables, Convertir los datos en un rango de valores  
from sklearn.preprocessing import StandardScaler
s_x = StandardScaler()
s_y = StandardScaler()
x = s_x.fit_transform(x)
y = s_y.fit_transform(y.reshape(-1,1))

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# Ajustar el svr con el dataset
reg_svr = SVR(kernel="rbf")
reg_svr.fit(x,y)

# predecir nueva variable segun el modelo
y_predict = reg_svr.predict(s_x.transform([[6.5]]))
y_predict = s_y.inverse_transform(y_predict)

# Graficar datos
#Crear lista de datos 
x_grid = np.arange(min(x), max(x), 0.1) 
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, reg_svr.predict(x_grid), color="blue")
plt.title("Modelo de regresino SVR")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# Graficar datos reales
#Crear lista de datos
plt.scatter(s_x.inverse_transform(x), s_y.inverse_transform(y), color="red")
plt.plot(s_x.inverse_transform(x_grid), s_y.inverse_transform(reg_svr.predict(x_grid)), color="blue")
plt.title("Modelo de regresino SVR")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


















