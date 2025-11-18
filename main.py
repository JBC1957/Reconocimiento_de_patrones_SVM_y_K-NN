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

df = pd.read_csv("ecommerce_dataset_10000.csv")

print("Primeras filas del conjunto de datos:")
print(df.head())

print("\nNombres de las columnas:")
print(df.columns)

# ==========================================
# 3. Elegir columna objetivo y limpiar
# ==========================================

# Columna que queremos predecir
columna_objetivo = "order_status"    # si quieres otra, cámbiala aquí

# Eliminar filas sin etiqueta
df = df.dropna(subset=[columna_objetivo])

# Columnas que NO queremos usar como entrada (IDs, texto largo, fechas)
columnas_a_eliminar = [
    "customer_id",
    "first_name",
    "last_name",
    "signup_date",
    "product_id",
    "product_name",
    "order_id",
    "order_date",
    "review_text",
    "review_id",
    "review_date",
]

# Nos quedamos sólo con las columnas útiles
X = df.drop(columns=[columna_objetivo] + columnas_a_eliminar)
y = df[columna_objetivo]

print("\nColumnas usadas como entrada (X):")
print(X.columns)

# Convertir categóricas a numéricas (one-hot)
X = pd.get_dummies(X, drop_first=True)

# Por si quedan valores perdidos, los eliminamos
datos = pd.concat([X, y], axis=1).dropna()
X = datos.drop(columns=[columna_objetivo])
y = datos[columna_objetivo]

print("\nTamaño final de los datos (filas, columnas):", X.shape)

# ==========================================
# 4. Separar entrenamiento / prueba y escalar
# ==========================================

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

escalador = StandardScaler()
X_entrenamiento_escalado = escalador.fit_transform(X_entrenamiento)
X_prueba_escalado = escalador.transform(X_prueba)

print("\nTamaño entrenamiento:", X_entrenamiento_escalado.shape)
print("Tamaño prueba:", X_prueba_escalado.shape)

# ==========================================
# 5. MODELO SVM (SVC)
# ==========================================

print("\n================= MODELO SVM (SVC) =================")

mejor_C = None
mejor_exactitud_svm = 0
mejor_modelo_svm = None

# Sólo kernel lineal (mucho más rápido) y varios C
lista_C = [0.1, 1, 10]

for C in lista_C:
    print(f"Entrenando SVM con C = {C} ...")
    svm = SVC(C=C, kernel="linear", random_state=0)

    svm.fit(X_entrenamiento_escalado, y_entrenamiento)
    predicciones = svm.predict(X_prueba_escalado)
    exactitud = accuracy_score(y_prueba, predicciones)

    print(f"   Exactitud = {exactitud:.4f}")

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

mejor_k = None
mejor_exactitud_knn = 0
mejor_modelo_knn = None

lista_k = [3, 5, 7, 9]

for k in lista_k:
    print(f"Entrenando k-NN con k = {k} ...")
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_entrenamiento_escalado, y_entrenamiento)
    predicciones = knn.predict(X_prueba_escalado)
    exactitud = accuracy_score(y_prueba, predicciones)

    print(f"   Exactitud = {exactitud:.4f}")

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

if mejor_exactitud_svm > mejor_exactitud_knn:
    print("\nEn este conjunto de datos, la SVM ha funcionado mejor que k-NN.")
elif mejor_exactitud_knn > mejor_exactitud_svm:
    print("\nEn este conjunto de datos, k-NN ha funcionado mejor que la SVM.")
else:
    print("\nEn este conjunto de datos, ambos modelos tienen una exactitud parecida.")
