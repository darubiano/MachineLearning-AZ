# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:09:36 2019

@author: darub
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# mostrar valores con 4 decimales
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

dataset = pd.read_csv("50_Startups.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# Codificar datos categoricos remplazar los paises por 0,1,2 etc.
from sklearn import preprocessing
#crear codificador de datos
le_x = preprocessing.LabelEncoder()
#transformar los paises por valores numericos (California 0, Florida 1 y New York 2)
x[:,3] = le_x.fit_transform(x[:,3])
# Proceso inverso
# x[:,3] = le_x.inverse_transform(list(x[:,3]))
print(x)


# dividir datos para entrenamiento y testeo, tomar 10 de 30 elementos para testear
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
regresion = LinearRegression()
regresion.fit(x_train, y_train)

# prediccion de los resultados del conjunto de testing
y_pred = regresion.predict(x_test)


# construir el modelo optimo ultilizando la eliminacion hacia atras
import statsmodels.api as sm
# agregar una columna de 1s, como coeficiente de termino independiente para saber que variables se pueden quitar
x = np.append(arr = np.ones((50, 1)).astype(int), values=x , axis = 1)
# variable de variables independientes optimas para predecir la variable dependiente
x_opt =x[:, [0,1,2,3,4]]
#valor de significacion
SL = 0.05
# minimos cuadrados, endog la que se desea predecir, exog 
regressor_OLS = sm.OLS(endog= y, exog= x_opt.astype(float)).fit()
# Devuelve los coeficientes del model y el P valor que se desea buscar menos a 0.05 P>|t|
print(regressor_OLS.summary())

# se elimina la columna con el p valor mas alto. en este caso la 4
x_opt = x[:, [0,1,2,3]].astype(float)
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())

# se elimina la columna con el p valor mas alto, en este caso 2
x_opt = x[:, [0,1,3]].astype(float)
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())

# se elimina la columna con el p valor mas alto, en este caso 3
x_opt = x[:, [0,1]].astype(float)
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())


# Nuevo modelo
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)
regresion = LinearRegression()
regresion.fit(x_train_o, y_train_o)
y_pred_o = regresion.predict(x_test_o)





