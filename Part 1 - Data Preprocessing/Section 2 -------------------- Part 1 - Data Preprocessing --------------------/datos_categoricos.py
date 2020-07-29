# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:23:15 2019

@author: darub
"""

#importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

# tomar valores independientes
x = dataset.iloc[:,:-1].values
# tomar valores dependientes
y = dataset.iloc[:,3].values

print(x)
print(y)
