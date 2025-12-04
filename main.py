'''
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 '''

# ==========================================
# 1. Importar librerías
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 2. Cargar datos
# ==========================================

# Cargamos el conjunto de datos original
df = pd.read_csv("ecommerce_dataset_10000.csv")

print("Primeras filas del conjunto de datos:")
print(df.head())

print("\nNombres de las columnas:")
print(df.columns)

# ==========================================
# 3. Elegir columna objetivo y limpiar
# ==========================================

# Columna que queremos predecir.
# IMPORTANTE: esta columna NO existe todavía en el CSV original.
# Aparece después de aplicar pd.get_dummies sobre la columna 'gender'.
# 'gender_Male' será 1 si el cliente es hombre y 0 en caso contrario.
columna_objetivo =  'country_USA' # "gender_Male"    # si quieres otra, cámbiala aquí

# Eliminamos cualquier fila que tenga algún valor perdido en cualquier columna.
# Es una decisión sencilla (en vez de imputar valores) que garantiza
# que trabajamos sólo con ejemplos completos.
df = df.dropna()

# Columnas que NO queremos usar como entrada porque:
# - Son identificadores (IDs) que no generalizan bien.
# - Son texto muy específico (nombres).
# - Son fechas que no estamos transformando a números.
columnas_a_eliminar = [
    "customer_id",
    "first_name",
    "last_name",
    "signup_date",
    "product_name",
    'product_id',
    "order_id",
    "order_date",
    "review_id",
    "review_date",
]

# Quitamos esas columnas del DataFrame
df = df.drop(columns=columnas_a_eliminar)

# Convertimos todas las columnas categóricas restantes en variables numéricas
# mediante one-hot encoding. Usamos drop_first=True para evitar colinealidad
# y reducir una columna por cada variable categórica.
df = pd.get_dummies(df, drop_first=True)

# Ahora que ya tenemos todas las columnas numéricas, separamos X e y.
# y será la columna 'gender_Male' (0/1), y X el resto de columnas.
X = df.drop(columns=[columna_objetivo])
y = df[columna_objetivo]

print("\nColumnas usadas como entrada (X):")
print(X.columns)

print("\nTamaño final de los datos (filas, columnas):", X.shape)

# ==========================================
# 4. Separar entrenamiento / prueba y escalar
# ==========================================

# Dividimos en entrenamiento (80%) y prueba (20%).
# Usamos stratify=y para mantener la proporción de hombres/no hombres
# en ambos subconjuntos.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# Escalamos las características numéricas para que todas tengan
# media 0 y desviación típica 1. Esto es importante tanto para SVM
# como para k-NN.
escalador = StandardScaler()
X_entrenamiento_escalado = escalador.fit_transform(X_entrenamiento)
X_prueba_escalado = escalador.transform(X_prueba)

print("\nTamaño entrenamiento:", X_entrenamiento_escalado.shape)
print("Tamaño prueba:", X_prueba_escalado.shape)

# ==========================================
# 5. MODELO SVM (SVC)
# ==========================================

print("\n================= MODELO SVM (SVC) =================")

mejor_C = None                # mejor valor de C encontrado
mejor_exactitud_svm = 0       # mejor exactitud obtenida con SVM
mejor_modelo_svm = None       # referencia al mejor modelo entrenado

# Probamos sólo kernel lineal (mucho más rápido que 'rbf' con muchas columnas)
# y varios valores del parámetro C para ajustar la regularización.
lista_C = [0.1, 1, 10]

for C in lista_C:
    print(f"Entrenando SVM con C = {C} ...")
    svm = SVC(C=C, kernel="linear", random_state=0)

    # Entrenamos con los datos de entrenamiento escalados
    svm.fit(X_entrenamiento_escalado, y_entrenamiento)

    # Predecimos sobre el conjunto de prueba
    predicciones = svm.predict(X_prueba_escalado)

    # Calculamos la exactitud (porcentaje de aciertos)
    exactitud = accuracy_score(y_prueba, predicciones)

    print(f"   Exactitud = {exactitud:.4f}")

    # Si este modelo es mejor que los anteriores, lo guardamos
    if exactitud > mejor_exactitud_svm:
        mejor_exactitud_svm = exactitud
        mejor_C = C
        mejor_modelo_svm = svm

print("\nMejor SVM:")
print("C =", mejor_C)
print("Exactitud en prueba =", mejor_exactitud_svm)

print("\nInforme de clasificación SVM:\n")
pred_mejor_svm = mejor_modelo_svm.predict(X_prueba_escalado)
print(classification_report(y_prueba, pred_mejor_svm))

# ==========================================
# 6. MODELO k-NN
# ==========================================

print("\n================= MODELO k-NN =================")

mejor_k = None                # mejor número de vecinos
mejor_exactitud_knn = 0       # mejor exactitud obtenida con k-NN
mejor_modelo_knn = None       # referencia al mejor modelo k-NN

# Probamos varios valores de k (número de vecinos)
lista_k = [3, 5, 7, 9]

for k in lista_k:
    print(f"Entrenando k-NN con k = {k} ...")
    knn = KNeighborsClassifier(n_neighbors=k)

    # Entrenamos con los datos de entrenamiento escalados
    knn.fit(X_entrenamiento_escalado, y_entrenamiento)

    # Predecimos sobre el conjunto de prueba
    predicciones = knn.predict(X_prueba_escalado)

    # Calculamos la exactitud
    exactitud = accuracy_score(y_prueba, predicciones)

    print(f"   Exactitud = {exactitud:.4f}")

    # Guardamos el modelo si mejora el resultado
    if exactitud > mejor_exactitud_knn:
        mejor_exactitud_knn = exactitud
        mejor_k = k
        mejor_modelo_knn = knn

print("\nMejor k-NN:")
print("k =", mejor_k)
print("Exactitud en prueba =", mejor_exactitud_knn)

print("\nInforme de clasificación k-NN:\n")
pred_mejor_knn = mejor_modelo_knn.predict(X_prueba_escalado)
print(classification_report(y_prueba, pred_mejor_knn))

# ==========================================
# 7. Resumen
# ==========================================

print("\n================= RESUMEN FINAL =================")
print(f"SVM -> mejor C = {mejor_C}, exactitud = {mejor_exactitud_svm:.4f}")
print(f"k-NN -> mejor k = {mejor_k}, exactitud = {mejor_exactitud_knn:.4f}")

# Comparación final de los dos clasificadores
if mejor_exactitud_svm > mejor_exactitud_knn:
    print("\nEn este conjunto de datos, la SVM ha funcionado mejor que k-NN.")
elif mejor_exactitud_knn > mejor_exactitud_svm:
    print("\nEn este conjunto de datos, k-NN ha funcionado mejor que la SVM.")
else:
    print("\nEn este conjunto de datos, ambos modelos tienen una exactitud parecida.")
