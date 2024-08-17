# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:03:13 2024

@author: ismac
"""

# Regresion con Arboles de Decision

# Como importar las librerias
import numpy as np    
import matplotlib.pyplot as plt    
import pandas as pd    


# Importar el dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values    
y = dataset.iloc[:, 2].values     


# Divirdir el dataset en conjunto de entrenamiento y conjunto de testing
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

                                                                                               
# Escalado de variables
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 


# Ajustar la regresion  con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)


# Prediccion de nuestros modelos
y_pred = regression.predict(6.5)


# Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "r")
plt.plot(X, regression.predict(X), color = "orange")
plt.title("Modelo de regresion polinomico")
plt.xlabel("Nivel del empleado")
plt.ylabel("Salario")
plt.show()





