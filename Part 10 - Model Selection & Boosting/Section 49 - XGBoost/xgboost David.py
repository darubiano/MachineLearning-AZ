# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:47:54 2020
conda install -c conda-forge xgboost
@author: darub
"""

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# cargar dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# dividir los datos
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Codificar lo datos
from sklearn import preprocessing
le_x1 = preprocessing.LabelEncoder()
le_x2 = preprocessing.LabelEncoder()
#transformar los paises por valores numericos (france 0, Germany 1 y Spain 2)
x[:,1]=le_x1.fit_transform(x[:,1])
# Tranformar el genero valores numericos (female 0, male 1)
x[:,2]=le_x2.fit_transform(x[:,2])


#Dividir el dataset en conjunto de entrenamiento y conjunto de testeo
from sklearn.model_selection import train_test_split#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#pip install xgboost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# predecir
y_pred = classifier.predict(x_test)


# Matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Representacion grafica de la matriz
import seaborn as sn
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,cmap='Blues')# font size
plt.title("Matriz de confusion")
plt.xlabel("Valores predichos")
plt.ylabel("Valores reales")
plt.show()


diagonal=0
for i in range(len(cm)):
    diagonal += cm[i][i]
accuracy = diagonal/cm.sum()
print("Accuracy: ",accuracy)


# Aplicar validacion cruzada
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()




