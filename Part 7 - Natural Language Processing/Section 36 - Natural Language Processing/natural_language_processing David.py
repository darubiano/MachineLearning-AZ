# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:11:39 2020

@author: darub
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

# quoting=3 ignorar comillas dobles "
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter= "\t", quoting=3)

#Limpieza de texto
import re
#import nltk
# Descargar Palabrar sobrantes articulos etc
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

texts = []

for i in range(0, dataset.shape[0]):
    
    # ^ elemento que no quiero eliminar letras mayusculas y minisculas de la a-z
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]).lower().split()
    
    # Reducir palabras loved a love etc
    ps = PorterStemmer()
    
    # Valida que la palabra no se encuentre en los stopwords
    review = [ps.stem(word) for word in review 
              if not word in set(stopwords.words('english'))]
    
    #juntar las palabras con un espacio en blanco
    review = ' '.join(review)
    texts.append(review)
    
# Crear el Bag of words (bolsa de palabras)
from sklearn.feature_extraction.text import CountVectorizer
# max_features=1500 maximas palabras frecuentes
cv = CountVectorizer(max_features=1500)
 # Matriz de caracteristicas
x = cv.fit_transform(texts).toarray()
y = dataset.iloc[:,-1].values


# Dividir dataset en conjunto de entranamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


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
plt.title("Matriz de confusion bayes")
plt.xlabel("Valores predichos")
plt.ylabel("Valores reales")
plt.show()

# Precision del modelo
from sklearn.metrics import classification_report
report_bayes = classification_report(y_test, y_pred)
print(report_bayes)

#Ajustar el modelo de SVM mas cercanos en el conjunto de entranamiento
from sklearn.svm import SVC
classifier_svm = SVC(kernel="rbf",random_state=0)
classifier_svm.fit(x_train, y_train)
# predecir
y_pred_svm = classifier_svm.predict(x_test)

# matriz de confusion
cm_svm = confusion_matrix(y_test,y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)
print(report_svm)


# Aplicar la mejora de Grid para optimizar el modelo
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C':[1, 10, 100, 1000],'kernel':['linear']},
    {'C':[1, 10, 100, 1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]},
    {'C':[1, 10, 100, 1000],'kernel':['poly']},
    {'C':[1, 10, 100, 1000],'kernel':['sigmoid']},
    ]

grid_search = GridSearchCV(estimator = classifier_svm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
#Encontrar los parametros
grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

