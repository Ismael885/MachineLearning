# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:02:27 2024

@author: ismac
"""

# Plantilla de Pre Procesado

# Como importar las librerias
import numpy as np    # Tratamiento de numeros en python
import matplotlib.pyplot as plt    # Representacion grafico python
import pandas as pd    # Carga y tratamiento de datos


# Importar el dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values    # Obtiene todas las filas y todas las columnas excepto la ultima
y = dataset.iloc[:, 3].values     # .values indica que quiero obtener solo los valores, no las posiciones ni demas





# Divirdir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # random_size porcentaje de datos para test
                                                                                               # random_state semilla aleat para la division del conjunto

                                                                                               
# Escalado de variables
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""         # Usamos transform para que se aplique el mismo escalado que al conjunto de entrenamiento

