# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:35:31 2020

@author: darub
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importar datos
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append(list(dataset.iloc[i,:].dropna()))

# Entrenar el algoritmo apriori
# pip install apyori
from apyori import apriori
rules = apriori(transactions, min_support=0.003 ,min_confidence=0.2 ,
                min_length = 2, min_lift = 3)

#Visualizacion de los resultados
rule = []
support = []
confidence = []
lift = []

for item in list(rules):
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    rule.append(items[0] + " -> " + items[1])
    #second index of the inner list
    support.append(str(item[1]))
    #third index of the list located at 0th
    #of the third index of the inner list
    confidence.append(item[2][0][2])
    lift.append(item[2][0][3])
 
output_ds  = pd.DataFrame({'rule': rule,
                           'support': support,
                           'confidence': confidence,
                           'lift': lift
                          }).sort_values(by = 'lift', ascending = False)








