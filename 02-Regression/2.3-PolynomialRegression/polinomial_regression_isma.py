# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:36:37 2024

@author: ismac
"""

# Regresion Polinimica

# Como importar las librerias
import numpy as np    
import matplotlib.pyplot as plt    
import pandas as pd    


# Importar el dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values    
y = dataset.iloc[:, 2].values     


# Divirdir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

                                                                                               
# Escalado de variables
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 

# Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizacion de los resultados del Modelo Lineal
plt.scatter(X, y, color = "r")
plt.plot(X, lin_reg.predict(X), color = "b")
plt.title("Modelo de regresion lineal")
plt.xlabel("Nivel del empleado")
plt.ylabel("Salario")
plt.show()


# Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "r")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "orange")
plt.title("Modelo de regresion polinomico")
plt.xlabel("Nivel del empleado")
plt.ylabel("Salario")
plt.show()


# Prediccion de nuestros modelos
lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))




