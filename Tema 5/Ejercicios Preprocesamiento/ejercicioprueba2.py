# (a) (0.4 puntos) Diseñe e implemente una estrategia de preprocesamiento 
# para variables numéricas, categóricas y ordinales. Debe incluir el 
# tratamiento de: valores perdidos, la conversión de total_charges a 
# numérico, outliers/asimetría en avg_call_duration, 
# alta cardinalidad / categorías raras. Justifique brevemente.

# Puesto que encontramos datos tanto categoricos como numericos dentro 
# del dataset necesitaremos separar estos de la variable objetivo asi 
# como tratarlos de forma diferente. Puesto que faltan datos en algunas 
# tablas deberemos imputar los mismos, datos categoricos mediante moda 
# y numericos mediante mediana. De igual manera para tratar de igual 
# manera los datos numericos deberemos estandarizarlos para que tengan 
# una media = 0, y desviacion = 1

x = churn_telco.drop(column = ["churn"]) # datos sin churn
y = churn_telco(column = ["churn"]) # solo columna churn

# Separamos los tipos de columnas para hacer sus transformaciones
#correspondientes.
# La variable total_charges viene como valor de texto y hay que 
# convertirlo a numerico por lo tanto se procede a su conversion de 
# los valores que se puedan convertir y aquellos que no se daran por 
# nulos para imputarlos mas tarde


columnas_categoricas = churn_telco["payment_method", "internet_service", "region"]
columnas_numericas = churn_telco["tenure_months", "monthly_charges", "support_tickets", "avg_call_duration", "total_charges"]


# Pasamos a imputar datos faltantes y estandarizar. Usaremos one-hot 
# encoder para los datos categoricos

# La variable avg_call_duration presenta asimetría y valores extremos. 
# Para evitar que los outliers dominen el modelo, se aplica una 
# transformación que reduce la asimetría y un escalado robusto.
pipelines_numericos = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", RobustEscaler())
])

# Las variables categóricas con alta cardinalidad contienen categorías 
# poco frecuentes. Estas se agrupan en una categoría común para evitar 
# un exceso de variables y reducir el sobreajuste.
# La variable contract es ordinal y se codifica respetando su orden 
# natural para no perder información.
pipelines_categoricos = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])



transformacion = ColumnTransformer([
    ("numericas", pipelines_numericos, columnas_numericas),
    ("categoricas", pipelines_categoricos, columnas_categoricas)
])



# (b) (0.3 puntos) Seleccione y configure un modelo apropiado para este 
# problema. Justifique su elección.


# Para este problema donde tenemos que conocer la probabilidad de
# pertenencia a una clase pudiendo expresarse como una probabilidad
# de 0 a 1, podemos usar una funcion sigmoide. Lo que nos lleva 
# a usar la regresion logistica, la cual es un clasificador que nos
# permite exactamente eso

modelo = LogisticRegression(max_iter = 1000, class_weight = "balanced")

final = Pipeline([
    ("procesamieto", transformacion),
    ("modelo", modelo)
])

# Class weight balanced para que los pesos se ajsuten si esque un grupo 
# tiene mauor representacion que otro en el dataset



# (c) (0.3 puntos) Seleccione una métrica de evaluación apropiada y 
# evalúe el modelo usando validación cruzada estratificada con 5 folds.
#  Justifique la métrica.

# Podemos elegir entre varias metricas de evaluacion como F1 o Auc-Roc
# sin embargo puesto que no tenemos un umbral establecido claramente
# y lo que queremos es hacer una clasificacion de clases resulta mas
# efectivo, ya que F1 sirve sin embargo solo para un umbral especifico

kfolds = StratifiedKFold(n_splits = 5, shuffle = True)

auc_roc_score = cross_val_score(final, x, y, cv = kfolds, scoring="auc-roc")