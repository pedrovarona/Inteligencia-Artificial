# a) Dise˜ne e implemente una estrategia de preprocesamiento para las 
# variables num´ericas y categ´oricas. Justifique brevemente sus 
# decisiones. 

# Puesto que tenemos variables categoricas y numericas las cuales
# no podemos tratarlas igual. Existen Nan dentro del dataset por lo
# que hay que imputar. Las variables categoricas las imputaremos por
# moda y las variables numericas por mediana. Por otro lado deberemos 
# estandarizar para que los valores se encuentren cerca de media = 0
# y sd = 1

# Comenzamos analizando x e y

x = wine.drop(columns = ["quality"]) # todas las variables que no son quality
y = wine["quality"] # columna quality


# Sacamos columnas categoricas y numericas para luego poder imputarlas

columna_categorica = ["color"]
columna_numerica = ["alcohol", "sulphates", "pH"]

# Imputamos y estandarizamos
pipeline_numericas = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

pipeline_categoricas = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("onehot", OneHotEncoder())
])


transformacion = ColumnTransformer([
    ("numericas", pipeline_numericas, columna_numerica),
    ("categoricas", pipeline_categoricas, columna_categorica)
])



# b)  Seleccione y configure un modelo apropiado para este problema. 
# Justifique su elecci´on.

# Seleccionamos el modelo de regresion logistica, es perfecto usar un 
# clasificador para este caso y al funcionar con la funcion sigmoide
# viene perfecto para el tipo de ejercicio que se busca, calcular
# probabilidades de pertenencia a las clases.
# Otros modelos serían menos optimos como arboles, svm, etc. Además
# teniendo ya los datos transformados nos viene bien.


modelo = LogisticRegression(max_iter = 1000, class_weight = "balanced")

final = Pipeline([
    ("procesamiento", transformacion),
    ("model", modelo)
])


# c) Seleccione una m´etrica de evaluaci´on apropiada y eval´ue el 
# modelo usando validaci´on cruzada con 5 folds. Justifique la elecci´on 
# de la m´etrica.

# Como metrica podemos usar tanto F1-score como Auc-roc. Elegire F1-Score
# ya que me permite tener en cuenta tanto el recall como precision por
# partes iguales, ya que tanto detectar un vino malo como bueno (falso positivo),
# asi como no detectar un buen vino (falso negativo) es igual de costoso.

kfold = StratifiedKFold(n_splits = 5, shuffle = True)

f1 = cross_val_score(final, x, y, cv = kfold, scoring = "f1")