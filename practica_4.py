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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


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

# In[ ]:


#Comprobamos que no hay valores ausentes en ninguna de las columnas.
ausentes=diamonds.isnull().sum()
ausentes


# In[ ]:


df = diamonds


# In[ ]:


#Para poder aplicar los distintos modelos las columnas cut, color y clarity deben ser expresadas numéricamente.
le = LabelEncoder()


# En las columnas 'cut' y 'clarity' decidimos no usar LabelEncoder porque al ordenar alfabéticamente las categorias perdemos el orden original.

# In[ ]:


df['cut'] = df['cut'].apply(lambda x: ['Fair','Good','Very Good', 'Premium', 'Ideal'].index(x))
print('Fair, Good, Very Good, Premium, Ideal se corresponden con los números del 0 al 4 respectivamente.')


# In[ ]:


df.color = le.fit_transform(diamonds.color)
print(list(le.classes_), 'se corresponden con los números del 0 al 6 respectivamente.')


# In[ ]:


df['clarity'] = df['clarity'].apply(lambda x: ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(x))
print('I1 (peor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (mejor) se corresponden con los números del 0 al 7 respectivamente.')


# In[ ]:


df.head()


# In[ ]:


fig, ([ax1,ax2,ax3],[ax4,ax5, ax6]) = plt.subplots(nrows = 2, ncols=3, figsize=(20,10))   

ax1.hist(df.cut, label=['0:Fair, 1:Good, 2:Very good, 3:Premium, 4:Ideal'], range = (-0.75,4.25))
ax1.set_xticks([0,1,2,3,4])
ax1.legend(loc=0)
ax1.set_title('Clasificación por corte')

ax2.hist(df.carat, bins = 15)
ax2.set_title('Clasificación por quilates')

ax3.hist(df.color, label=['0:Mejor color, 6:Peor color'], bins = 14, range = (-0.75,6.25))
ax3.set_xticks([0,1,2,3,4,5,6])
ax3.legend(loc=0)
ax3.set_title('Clasificación por color')

ax4.hist(df.clarity, label=['0:Peor claridad, 7:Mejor claridad'], bins = 16, range = (-0.75,7.25))
ax4.set_xticks([0,1,2,3,4,5,6,7])
ax4.legend(loc=0)
ax4.set_title('Clasificación por claridad')
                            
ax5.hist(df.depth, bins=15)
ax5.set_title('Clasificación por altura relativa')
                            
ax6.hist(df.table, bins=15)
ax6.set_title('Clasificación por ancho relativo');


# In[ ]:


#A continuación mostramos los principales estadísticos de las variables
df.describe()


# In[ ]:


#Veamos, para cada tipo de corte, la media de las características cuantitativas de los diamantes.
df_res = pd.pivot_table(df, values=['carat', 'depth', 'table', 'x','y','z'], index='cut', aggfunc=np.mean)
df_res['']=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'] 
#añadimos una columna con las cateogrías de calidad de corte correspodientes a cada número

cols = df_res.columns.tolist()
cols = cols[-1:] + cols[:-1] #ponemos la columna al lado de cut

df_res[cols] 


# In[ ]:


#Separamos la columna correspondiente a la variable dependiente de las explicativas
y = df.iloc[:,1]
X = df.drop(['cut','price'], axis=1)


# Vamos a seleccionar un conjunto 'equilibrado' (mediante el stratify) para, más tarde, analizar las predicciones que hace el modelo sobre él. 
# 
# ¿Por qué 'equilibrado'? En la representación gráfica vimos que los cortes Fair y Good son menos frecuentes. Por tanto, si seleccionamos un fragmendo cualquiera podrían no aparecer. Como ejemplo parece más interesante evaluar el modelo en un dataframe que contenga todos los tipos de corte.

# In[ ]:


from sklearn.model_selection import train_test_split

X, X_2, y, y_2 = train_test_split(X,y, test_size = 0.1, stratify=y)


# Separamos en train y test para un análisis preliminar de los resultados de los modelos.

# In[ ]:


X_train, X_test, y_train, y_test =train_test_split(X,y)


# ### 2. Uso y validación de modelos
# 
# Bansádonos ahora en ```scikit-learn``` comparamos cuatro modelos de clasificación distintos para predecir la calidad del corte en función de las demás variables (excepto el precio):
# 
# * Decision Tree Classifier
# * Random Forest Classifier
# * Gradient Boosting Classifier
# * MLP Classifier
# 
# En primer lugar probaremos los cuatro modelos de forma preliminar y, según las precisiones obtenidas, haremos una validación cruzada en más profundidad en los modelos que parezcan ser más fiables.
# 
# Decidimos, porque creemos más conveniente, no reducir la dimensionalidad de los datos ni realizar un escalado.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model =DecisionTreeClassifier()
model.fit(X_train, y_train)

model.score(X_test,y_test)


# RandomForestClassifier es un "meta-estimador" que se ajusta a varios árboles de clasificación en varias submuestras del data frame y utiliza el promedio para mejorar la precisión predictiva y controlar el sobreajuste.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

model.score(X_test,y_test)


# GradientBoostingClassifier construye un modelo *forward* aditivo por etapas; permite la optimización de la función de pérdida. En cada etapa, los árboles de regresión se ajustan en el gradiente negativo de la función de pérdida.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

model.score(X_test,y_test)


# Multi-layer Perceptron classifier optimiza la función de pérdida usando como solver LBFGS, stochastic gradient descent o adam (adam por defecto).

# In[ ]:


from sklearn.neural_network import MLPClassifier #Redes neuronales

model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)

model.score(X_test,y_test)


# Según las accuracies obtenidas, decidimos analizar más detenidamente la aplicación del Random Forest Classifier [0.783] y el Gradient Boosting Classifier [0.767].
# 
# Observación: durante la creación del código este proceso se ha realizado varias veces con distintos conjuntos de train y test (en concreto, 7) y siempre se ha obtienido el mismo orden de los modelos según la accuracy. El modelo que más variaciones ha experimentado es la red neuronal (MLP Classifier), habiendo obtenido una accuracy máxima de 0.722 y una mínima de 0.581. No sabemos por qué funciona peor que los demás modelos, ya que las NN actúan como cajas negras. Quizá ajustando los parámetros de la red podría mejorarse algo el resultado, sin embargo, creemos más razonable centrarnos en los dos modelos anteriormente mencionados.

# ### Ajuste de parámetros
# #### Random Forest Classifier:

# In[ ]:


model = RandomForestClassifier(n_jobs=-1)

grid = {'max_depth': [5,10,20,50], "n_estimators": [200, 300, 400]}

cv = GridSearchCV(estimator=model, param_grid=grid, scoring='accuracy',cv=10, n_jobs=-1)

cv.fit(X, y)

cv.best_score_, cv.best_params_


# #### Gradient Boosting Classifier:

# In[ ]:


model = GradientBoostingClassifier()

grid = {'max_depth':[3,5,8], "n_estimators":[100, 200]}

#El tiempo de ejecución es muy alto (aún usando los dos núcleos del procesador),
#por lo que decidimos no crear un grid excesivamente grande. 

cv = GridSearchCV(estimator=model, param_grid=grid, scoring='accuracy',cv=10, n_jobs=-1)

cv.fit(X, y)

cv.best_score_, cv.best_params_


# Aunque para ambos modelos obtenemos resultados parecidos, Gradient Boosting Classifier alcanza la mayor precisión [0.798] entre todas las posibilidades analizadas para un máximo de 8 nodos en el árbol y un número de 100 estimadores.

# In[ ]:


cv.best_estimator_


# ## Ejemplo:
# Veamos ahora qué tal predice el mejor modelo encontrado con GridSearchCV para el ejemplo anteriormente seleccionado.
# 
# * Recordemos que 'Fair', 'Good', 'Very Good', 'Premium' y 'Ideal' se corresponden con los números del 0 al 4 respectivamente. 

# In[ ]:


y_pred = cv.predict(X_2)


# In[ ]:


cv.score(X_2, y_2) #Obtenemos una accuracy buena (en relación a los resultados que hemos ido teniendo)


# In[ ]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_2, y_pred))
print('')
print('Fair, Good, Very Good, Premium, Ideal')


# In[ ]:


my_array = confusion_matrix(y_2, y_pred)


# Como no es una variable binaria no hay un comando de python que extraiga directamente falsos negativos, falsos positivos, etc., por lo que lo programamos.

# In[ ]:


#Verdaderos cortes tipo j mal clasificados como i
cortes = ['Fair','Good','Very Good', 'Premium', 'Ideal'] 
print('False discovery rate:') #Tasa de error tipo I
for j in range(0,5):
    print('El porcentaje de diamantes de tipo', cortes[j], 'mal clasificados es del', 1 - my_array[j,j]/my_array[:,j].sum())


# In[ ]:


#Verdaderos cortes tipo mal clasificados como i
print('False omission rate:') #Tasa error tipo II
for i in range(0,5):
    print('El porcentaje de diamantes mal clasificados como',cortes[i],'es del', 1 - my_array[i,i]/my_array[i,:].sum())


# Las calidades que clasifica mejor son 'Fair' e 'Ideal', es decir, los dos extremos. Por otro lado, podemos ver que el modelo confunde más frecuentemente las calidades medias tanto como falsos negativos como falsos positivios (mirando dos a dos los tipos de corte), principalmente si son calidades contiguas. En total, la calidad que peor clasifica el modelo es 'Very Good', es decir, la calidad media, que incluso confunde en no pocas ocasiones con la máxima calidad, 'Ideal' (vemos que tiene una *false discovery rate* de 30% y una *false omission rate* del 40%).

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




