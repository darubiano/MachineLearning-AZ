# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:36:11 2020

GRID search

@author: darub
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state = 0)

# Normalizar datos
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Ajustar el modelo de SVM mas cercanos en el conjunto de entranamiento
from sklearn.svm import SVC
classifier = SVC(kernel="rbf",random_state=0)
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

# Aplicar la mejora de Grid para optimizar el modelo
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C':[1, 10, 100, 1000],'kernel':['linear']},
    {'C':[1, 10, 100, 1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]},
    {'C':[1, 10, 100, 1000],'kernel':['poly']},
    {'C':[1, 10, 100, 1000],'kernel':['sigmoid']},
    ]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
#Encontrar los parametros
grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_


















