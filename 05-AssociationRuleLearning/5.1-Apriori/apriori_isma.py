# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:49:24 2024

@author: ismac
"""

# Apriori 

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    

# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizacion de los resultados
results = list(rules)

results[0]