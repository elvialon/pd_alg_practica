#!/usr/bin/env python
# coding: utf-8

# In[ ]:


nombre = "Elvira Alonso González"
if nombre == "":
    print("Rellena tu nombre completo!")
else:
    print("Gracias", nombre, ":)")


# In[ ]:


#Librerías:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Práctica 4: dataset libre
# 
# 

# En esta práctica tendrás que elegir tu propio conjunto de datos y realizar una tarea de clasificación o regresión.
# 
# Para ello, se plantean distintas fases que puedes utilizar como pautas a la hora de realizarla.
# 
# Aquí van algunas referencias donde se pueden encontrar datasets:
# 
# * https://www.kaggle.com/datasets 
# 
# * https://archive.ics.uci.edu/ml/index.php
# 
# * https://datasetsearch.research.google.com
# 
# * También se puede usar algún dataset de otra asignatura (pero no visto en AMUL!)
# 
# * O generarlo vosotros mismos (por ej. a partir de sensores/wearables, señal GPS al salir a correr, gastos e ingresos bancarios etc etc)
# 
# Por normal general, deberían tener más de 1000 observaciones.
# 
# El resultado de esta fase debería ser uno o varios .csv (u otro formato como excel) con todos los datos que usarás.

# ### 0. Elección de dataset
# 
# Para este trabajo hemos elegido el data set Diamonds (https://www.kaggle.com/shivam2503/diamonds). Es un conjunto de datos clásico que contiene observaciones de distintos atributos de casi 54,000 diamantes.
# 
# Las distintas variables que encontraremos en el dataset son:
# 
# * price: precio en dólares (de 326 a 18,823)
# 
# * carat: quitales del diamante (de 0.2 a 5.01)
# 
# * cut: calidad del corte (Fair, Good, Very Good, Premium, Ideal)
#  
# * color: color del diamante, de J (el peor) a D (el mejor)
# 
# * clarity: medida de la claridad del diamante (I1 (peor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (mejor))
# 
# * depth: altura del diamante expresada como porcentaje del diámtro medio (de 43 a 79)
# 
# * table: mayor ancho del diamante expresado como un porcentaje de su diámetro promedio (de 43 a 95)
# 
# * x: altura in mm (de 0 a 10.74)
# 
# * y: ancho in mm (de 0 a 58.9)
# 
# * z: profundidad in mm (de 0 a 31.8)
# 
# 
# Este dataset es interesante tanto para regresión (p.e. sobre el precio) como para clasificación. En nuestro caso realizaremos un análisis de clásificación según la variable 'cut', es decir, veremos si las demás carácterísticas del diamante condicionan la calidad del corte que se le realice. Debido a que el precio es una carácterística posterior (está determinada por la cualidades generales del diamante, pero no al revés) parece razonable eliminarla del análisis. Por otra parte, las variables 'depth' y 'table' son interacciones **no lineales** de 'x', 'y' y 'z', por lo que conservaremos todas ellas. 
# 

# In[ ]:


#Cargamos el csv
diamonds = pd.read_csv("diamonds.csv", nrows=53940)


# In[ ]:


diamonds.columns


# In[ ]:


#Eliminamos la columna de índices que venía en el csv
diamonds = diamonds.drop('Unnamed: 0',axis=1)


# In[ ]:


diamonds.head() #depth y table son interaciones no lineales de x,y,z


# ### 1. Exploración y tratamiento de datos

# Basándote en las librerías ```numpy```, ```pandas``` y ```matplotlib``` haz las operaciones que creas convenientes y te ayuden a comprender mejor tu dataset y la tarea de clasificación/correspondiente. Para ello, algunos tipos de cosas que puedes poner:
# 
# * Operaciones como estadísticos o filtrados/groupby/merge de pandas para calcular información de interés.
# 
# * Visualizaciones con matplotlib de alguna(s) columna(s).
# 
# * "Arreglo" de datos: imputación de missings y outliers, calcular nuevas columnas a partir de las originales, etc.
# 
# El resultado de esta fase debería ser la definición de un conjunto de columnas que usarás como features (X) y otra columna que usarás como variable a predecir (y).

# ### 2. Uso y validación de modelos
# 
# Basándote ahora en ```scikit-learn``` aplica los modelos que consideres oportunos para ver qué tal predicen.
# Un esquema podría ser el siguiente:
# 
# * Probar tres modelos distintos de forma preliminar.
# 
# * Para el mejor de de los anteriores, hacer ya una CV más en profundidad con búsqueda de hiperparámetros para tratar de mejorarlo.
# 
# * Si lo consideras razonable, puedes hacer reescalados, reducción de dimensionalidad etc.
# 
# * Opcionalmente puedes usar modelos no vistos en la asignatura, la lista completa está aquí https://scikit-learn.org/stable/supervised_learning.html
# 
# * Aunque la extensión de lo anterior es variable: si tus datos son muy fáciles de tratar, puedes profundizar más en esta fase de modelos. Si tus datos son costosos de preprocesar, puedes reducir el número de modelos de esta fase..
# 
# * Aparte de evaluar alguna métrica, también puedes seleccionar o construir ejemplos concretos y examinar sus predicciones.
# 
# El resultado de esta fase debería ser unas estimaciones de las métricas que creas convenientes para cada configuración de los modelos. También puedes hacer visualizaciones sobre las métricas.

# ### 3. Conclusiones
# 
# Incluye un párrafo describiendo lo anterior y la utilidad (o no) de los modelos para tus datos en cuestión, y sus posibles aplicaciones al mundo real.

# ### Valoración
# 
# Esta práctica se valorará con un máximo de 10 puntos, teniendo un peso igual que las tres anteriores (cada práctica es por tanto el 10% de la nota).
# 
# Para la valoración, se tendrá en cuenta:
# 
# * Corrección y coherencia del código al dataset seleccionado.
# 
# * Simplicidad del código (evita bucles innecesarios, utiliza funciones ya implementadas en las librerías en vez de rehacerlas a mano, etc).
# 
# * Comentarios que expliquen cada paso que haces.

# In[ ]:




