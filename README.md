#Fichero markdown para el readme del proyecto

# Proyecto 4: dataset libre

**Enunciado de la práctica:**

En esta práctica tendrás que elegir tu propio conjunto de datos y realizar una tarea de clasificación o regresión.

Para ello, se plantean distintas fases que puedes utilizar como pautas a la hora de realizarla.

### 0. Elección de dataset

Aquí van algunas referencias donde se pueden encontrar datasets:

* https://www.kaggle.com/datasets 

* https://archive.ics.uci.edu/ml/index.php

* https://datasetsearch.research.google.com

* También se puede usar algún dataset de otra asignatura (pero no visto en AMUL!)

* O generarlo vosotros mismos (por ej. a partir de sensores/wearables, señal GPS al salir a correr, gastos e ingresos bancarios etc etc)

Por normal general, deberían tener más de 1000 observaciones.

El resultado de esta fase debería ser uno o varios .csv (u otro formato como excel) con todos los datos que usarás.

### 1. Exploración y tratamiento de datos

Basándote en las librerías ```numpy```, ```pandas``` y ```matplotlib``` haz las operaciones que creas convenientes y te ayuden a comprender mejor tu dataset y la tarea de clasificación/correspondiente. Para ello, algunos tipos de cosas que puedes poner:

* Operaciones como estadísticos o filtrados/groupby/merge de pandas para calcular información de interés.

* Visualizaciones con matplotlib de alguna(s) columna(s).

* "Arreglo" de datos: imputación de missings y outliers, calcular nuevas columnas a partir de las originales, etc.

El resultado de esta fase debería ser la definición de un conjunto de columnas que usarás como features (X) y otra columna que usarás como variable a predecir (y).

### 2. Uso y validación de modelos

Basándote ahora en ```scikit-learn``` aplica los modelos que consideres oportunos para ver qué tal predicen.
Un esquema podría ser el siguiente:

* Probar tres modelos distintos de forma preliminar.

* Para el mejor de de los anteriores, hacer ya una CV más en profundidad con búsqueda de hiperparámetros para tratar de mejorarlo.

* Si lo consideras razonable, puedes hacer reescalados, reducción de dimensionalidad etc.

* Opcionalmente puedes usar modelos no vistos en la asignatura, la lista completa está aquí https://scikit-learn.org/stable/supervised_learning.html

* Aunque la extensión de lo anterior es variable: si tus datos son muy fáciles de tratar, puedes profundizar más en esta fase de modelos. Si tus datos son costosos de preprocesar, puedes reducir el número de modelos de esta fase..

* Aparte de evaluar alguna métrica, también puedes seleccionar o construir ejemplos concretos y examinar sus predicciones.

El resultado de esta fase debería ser unas estimaciones de las métricas que creas convenientes para cada configuración de los modelos. También puedes hacer visualizaciones sobre las métricas.

### 3. Conclusiones

Incluye un párrafo describiendo lo anterior y la utilidad (o no) de los modelos para tus datos en cuestión, y sus posibles aplicaciones al mundo real.

### Valoración

Esta práctica se valorará con un máximo de 10 puntos, teniendo un peso igual que las tres anteriores (cada práctica es por tanto el 10% de la nota).

Para la valoración, se tendrá en cuenta:

* Corrección y coherencia del código al dataset seleccionado.

* Simplicidad del código (evita bucles innecesarios, utiliza funciones ya implementadas en las librerías en vez de rehacerlas a mano, etc).

* Comentarios que expliquen cada paso que haces.


