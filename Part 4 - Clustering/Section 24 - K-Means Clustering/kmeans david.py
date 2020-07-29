# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:44:31 2020

@author: darub
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importar datos
dataset = pd.read_csv('Mall_Customers.csv')
# Seleccionar datos a agrupar salario y gastos por cliente
x = dataset.iloc[:,[3,4]].values

#Metodo del codo para averiguar el numero optimo de clusters
from sklearn.cluster import KMeans
wcss = []

# Probar con 10 clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS (k)")
plt.show()


#Aplicar el metodo optimo 5 clusters
kmeans_op = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_op.fit(x)

# Predecir cada dato a que cluster pertenece
y_kmeans = kmeans_op.predict(x)


#Visualizacion de los clusters
#Ajustar imagen generada
fig, ax = plt.subplots(figsize=(7,7))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c="red", label="Cluster 1")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c="blue", label="Cluster 2")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c="green", label="Cluster 3")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c="cyan", label="Cluster 4")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c="magenta", label="Cluster 5")
#Baricentros
plt.scatter(kmeans_op.cluster_centers_[:,0], kmeans_op.cluster_centers_[:,1], s = 300, c="yellow", label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de Gastos (1-100)")
plt.legend()
plt.show()


