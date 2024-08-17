# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:37:07 2024

@author: ismac
"""

# Plantilla de Pre Procesado - Datos categoricos

# Como importar las librerias
import numpy as np    # Tratamiento de numeros en python
import matplotlib.pyplot as plt    # Representacion grafico python
import pandas as pd    # Carga y tratamiento de datos

# Importar el dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values    # Obtiene todas las filas y todas las columnas excepto la ultima
y = dataset.iloc[:, 3].values     # .values indica que quiero obtener solo los valores, no las posiciones ni demas

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])     # Para variables dummy
X = onehotencoder.fit_transform(X).toarray()      # No indicamos la columna a modificar porque lo indicamos arriba

labelencoder_y = LabelEncoder()     # Para la variable purchase
y = labelencoder_y.fit_transform(y)