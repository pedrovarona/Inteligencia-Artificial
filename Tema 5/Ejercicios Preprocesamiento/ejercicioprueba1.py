# a) Diseñe e implemente una estrategia de preprocesamiento para 
# las variables numéricas y categóricas. Justifique brevemente sus 
# decisiones.


# Puesto que en el dataset encontramos diferentes columnas con tipos
# distintos de informacion (categorica y numerica) deberemos tratar
# estos datos de forma diferente, es decir que los datos numericos
# sirvan para nuestro modelo por lo que habra que transformarlos.
# De igual manera necesitaremos estandarizar los datos numericos.
# Para solucionar el problema de los NaN deberemos imputar a las
# variables numericas por la mediana y a las categoricas por la
# moda.

# Detectamos x e y

x = customers.drop(column = ["churn"]) # datos sin churn
y = customers["churn"] #columna churn

# Conseguimos las columnas segun su tipo para usarlo mas tarde
columnas_numericas = customers["age", "monthly_fee", "tenure"]
columnas_categoricas = customers["contract_type", "internet_service", "support_calls"]


# Procedemos a imputar y estandarizar

pipelines_numericos = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", SimpleScaler())
])

pipelines_categoricos = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("onehot", OneHotEncoder())
])


transformacion = ColumnTransformer([
    ("numericas", pipeline_numericos, columnas_numericas),
    ("categoricas", pipelines_categoricos, columnas_categoricas)
])


# b) Seleccione y configure un modelo de aprendizaje automático 
# apropiado para este problema. Justifique su elección frente a otras 
# posibles alternativas.


# Puesto que la columna de la variable es binaria, un modelo muy util
# puede ser regresion logistica, perfecto para un entrenamiento de un
# modelo que debe aprender a agrupar por clases distintas. Por lo tanto
# un modelo que utilize como funcion sigmoide es perfecto para este caso.

modelo = LogisticRegression(max_iter = 1000, class_weight = "balanced")

final = Pipeline([
    ("procesamiento", transformacion),
    ("model", modelo)
])

# Creamos el modelo con class_weight balanced para que el modelo compense
# los pesos de las clases si esque una tiene mayor representacion en el
# dataset



# c) Seleccione una métrica de evaluación adecuada y evalúe el modelo 
# usando validación cruzada con 5 folds. Justifique la elección de la 
# métrica.


# Puesto que queremos evaluar un modelo que debe poder agrupar bien
# dentro de las diferentes clases (0 o 1) necesitamos una metrica como
# f1-score o AUC-ROC. Me decanto por F1-score ya que nos permite una
# evaluacion equilibrada entre recall y precision.

kfolds = StratifiedKFold(n_splits = 5, shuffle = True)

f1_score = cross_val_score(final, x, y, cv = kfolds, scoring = "f1")