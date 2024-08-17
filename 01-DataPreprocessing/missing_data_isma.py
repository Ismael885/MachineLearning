# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:38:00 2024

@author: ismac
"""

# Plantilla de Pre Procesado - Datos faltantes

# Como importar las librerias
import numpy as np    # Tratamiento de numeros en python
import matplotlib.pyplot as plt    # Representacion grafico python
import pandas as pd    # Carga y tratamiento de datos

# Importar el dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values    # Obtiene todas las filas y todas las columnas excepto la ultima
y = dataset.iloc[:, 3].values     # .values indica que quiero obtener solo los valores, no las posiciones ni demas

# Tratamiento de los NAs
from sklearn.preprocessing import Imputer    # Clase para el tratamiento de los NANs
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])    # Tomamos las columnas 1 y 2, 3 se pone porque no se toma el limite superior
X[:, 1:3] = imputer.transform(X[:, 1:3])