# 🛒 Clasificación de Datos de Ecommerce con SVM y k-NN

Proyecto en Python — Machine Learning con Scikit-Learn

## 📌 Descripción del proyecto

Este proyecto entrena dos clasificadores de Machine Learning —SVM (Máquina de Soporte Vectorial) y k-NN (k vecinos más próximos)— para predecir una columna objetivo derivada del conjunto de datos de pedidos de una tienda online.
En este caso, la columna aprendida es gender_Male, que indica si el cliente es hombre o no.

El código realiza todo el flujo completo de análisis y aprendizaje:

- Carga y exploración inicial del dataset.
- Limpieza y transformación de datos:
- Eliminación de columnas irrelevantes (IDs, nombres, fechas).
- Codificación one-hot de variables categóricas.
- Normalización mediante StandardScaler.
- División en entrenamiento y prueba.
- Entrenamiento y ajuste de parámetros de SVM y k-NN.
- Evaluación detallada mediante accuracy, precision, recall y f1-score.
- Comparación final de ambos modelos.

## ▶️ Cómo ejecutar el proyecto
### 1. Instala las dependencias
```bash
pip install pandas scikit-learn
```
### 2. Ejecuta el script
```bash
python main.py
```
Asegúrate de que ecommerce_dataset_10000.csv está en la misma carpeta.
## 📊 Resultados que muestra el programa

Al ejecutarlo, el código imprime:

### 1. Información del dataset
- Primeras filas.
- Nombres de columnas.
- Número de filas y columnas finales después del preprocesado.
### 2. Evaluación del modelo SVM
Para cada valor de C:
```python
Entrenando SVM con C = 0.1 ...
   Exactitud = 0.6895
```
Incluye un informe de clasificación:
```sql
precision | recall | f1-score | support
```
### 3. Evaluación del modelo k-NN
Para cada valor de k:
```python
Entrenando k-NN con k = 9 ...
   Exactitud = 0.6600
```
También se muestra un informe de métricas.
### 4. Comparación final
Ejemplo:
```pyhon
SVM -> mejor C = 0.1, exactitud = 0.6895
k-NN -> mejor k = 9, exactitud = 0.6600
```
## 🧠 Breve explicación de los resultados
- La SVM obtiene mejor rendimiento global (≈ 69 % de acierto) comparado con k-NN (≈ 66 %).
- La SVM predice mejor tanto la clase mayoritaria como la minoritaria, mientras que k-NN tiene dificultades para identificar correctamente a la clase “True”.

Esto muestra que, para este dataset, una SVM lineal es más adecuada que un modelo k-NN.
