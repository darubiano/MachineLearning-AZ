# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:45:58 2019

@author: darub
"""

#importar librerias
import numpy as np
import pandas as pd


dataset = pd.read_csv('Data.csv')

# tomar valores independientes
x = dataset.iloc[:,:-1].values
# tomar valores dependientes
y = dataset.iloc[:,3].values

print(x)
print(y)

# Tratamiento de los NaN
from sklearn.impute import SimpleImputer
# remplazar por la media los valores NaN 
imputer = SimpleImputer(strategy="mean")
#sacar la media de la edad y salario
# remplazar los valoes nan
x[:,1:3] = imputer.fit_transform(x[:,1:3])
print(x)

# Codificar datos categoricos remplazar los paises por 0,1,2 etc.
from sklearn import preprocessing
#crear codificador de datos
le_x = preprocessing.LabelEncoder()
#transformar los paises por valores numericos (france 0, Germany 1 y Spain 2)
x[:,0] = le_x.fit_transform(x[:,0])
print(x)
#x[:,0] = le_x.inverse_transform(list(x[:,0]))
#print(x)

from sklearn import preprocessing
#crear codificador de datos
le_y = preprocessing.LabelEncoder()
# transformar yes, no por 1, 0
y = le_y.fit_transform(y)
print(y)


"""Dividir el dataset en conjunto de entrenamiento y conjunto de testeo"""
from sklearn.model_selection import train_test_split
# x ,y porcentaje de testeo 20% y semilla aleatoria random
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)


# Escalado de variables (normalizacion de datos cuando sus valores son muy lejanos edad salario)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



# proceso inverso
# Descalar y transformar el resultado en tipo array object 
x_origin = np.asarray(sc_x.inverse_transform(x_test), dtype=np.object)

# decodificar paises
# convetir la columna paises en int
x_paises = list(map(int,x_origin[:,0]))
# converitr valores numericos a string nombre de paises
x_origin[:,0] = le_x.inverse_transform(x_paises)



























