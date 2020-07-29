# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:24:03 2020

@author: darub
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar los datos
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values


# Utilizar el dendrograma para encontrar el numero optimo de clusters
import scipy.cluster.hierarchy as sch
fig, ax = plt.subplots(figsize=(20,20))
dendrograma = sch.dendrogram(sch.linkage(x, method='ward', metric='euclidean'))
plt.title("Dendrograma")
plt.xlabel('Clientes')
plt.ylabel('Distancia Euclidea')
plt.show()


#Ajustar el clustering jerarquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)


#Visualizacion de los clusters
#Ajustar imagen generada
fig, ax = plt.subplots(figsize=(7,7))
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c="red", label="Cluster 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c="blue", label="Cluster 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c="green", label="Cluster 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c="cyan", label="Cluster 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c="magenta", label="Cluster 5")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de Gastos (1-100)")
plt.legend()
plt.show()

