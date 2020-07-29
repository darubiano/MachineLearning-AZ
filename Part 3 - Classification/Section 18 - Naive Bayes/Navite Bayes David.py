# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:49:45 2020

@author: darub
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar el data set
dataset = pd.read_csv("Social_Network_Ads.csv")

# datos dependientes y datos independientes
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Dividir dataset en conjunto de entranamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Ajustar el clasificador naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

# predecir datos
y_pred = classifier.predict(x_test)

# matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# Representacion grafica de la matriz
import seaborn as sn
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,cmap='Blues')# font size
plt.title("Matriz de confusion")
plt.xlabel("Valores predichos")
plt.ylabel("Valores reales")
plt.show()

# Representacion grafica de los resultado del algoritmo conjunto de entrenamiento
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
# contourf pintar todo el plano
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes rbf (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# Representacion grafica de los resultado del algoritmo de testing
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes rbf (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()










