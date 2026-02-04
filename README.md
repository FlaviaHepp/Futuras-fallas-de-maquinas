# Futuras-fallas-de-maquinas
Predicción de fallas de máquinas por medio de datos obtenidos de sensores

Acerca de los datos
El conjunto de datos incluye lecturas de sensores relacionados con las máquinas, como:
Índice de calidad del aire (AQ).
Niveles de compuestos orgánicos volátiles (VOC).
Temperatura de funcionamiento (Temperature).
Presión de entrada (IP),
También incluye una columna binaria (fail) que
Pasos principales en el archivo
Carga y exploración de datos:

Se cargaron los datos desde un archivo CSV (datamaq.csv).
Se id
Se creó una matriz de correlación para identificar relaciones entre variables.
Selección de características:

Según la matriz de correlación, se identificaron AQ y VOC como las variables más relevantes para predecir fallas.
Entrenamiento y prueba de modelos:

Los datos se dividieron en conjuntos de entrenamiento (80%) y prueba (20%).
Se probaron diferentes modelos de clasificación:
Redes neuronales (TensorFlow).
XGBoost.
Random Forest.
Gradient Boosting.
AdaBoost.
Extra Trees.
Árboles de decisión.
Se evaluaron los modelos utilizando métricas como la precisión (accuracy), matriz de confusión, y el informe de clasificación.
Optimización de hiperparámetros:

Se utilizaron búsquedas en cuadrícula (GridSearchCV) para encontrar los mejores hiperparámetros para los modelos, como:
Número de estimadores (n_estimators).
Tasa de aprendizaje (learning_rate).
Profundidad máxima del árbol (max_depth).
Resultados del modelo:

Los modelos fueron evaluados con precisión (accuracy), donde se reportaron los mejores hiperparámetros y puntuaciones.
Conclusiones
Predicción de fallas:
Los modelos entrenados pueden predecir fallas con un nivel razonable de precisión, lo que permite anticiparse y realizar mantenimiento preventivo.
Modelos destacados:
El modelo XGBoost y el Random Forest demostraron ser altamente efectivos en este caso.
Importancia de los sensores:
Las lecturas de calidad del aire (AQ) y compuestos volátiles (VOC) son indicadores clave para identificar problemas en las máquinas.
Aplicaciones prácticas
Este análisis es útil para la mantenimiento predictivo, ayudando a reducir tiempos de inactividad y costos asociados a fallas inesperadas.
Los resultados pueden aplicarse en sistemas de monitoreo industrial para mejorar la confiabilidad y eficiencia operativa.

# Predicción de fallas de máquinas usando datos de sensores

Este proyecto desarrolla un sistema de **predicción de fallas de máquinas (predictive maintenance)** utilizando datos de sensores industriales y técnicas de **machine learning supervisado**.

El objetivo es **anticipar fallas antes de que ocurran**, permitiendo reducir tiempos de inactividad, costos de mantenimiento y riesgos operativos.

---

## 🏭 Contexto del problema

En entornos industriales, las fallas inesperadas de maquinaria generan:
- pérdidas económicas
- interrupciones en la producción
- riesgos de seguridad

El uso de **datos de sensores en tiempo real** permite detectar patrones anómalos y estimar la probabilidad de falla de una máquina con anticipación.

Este proyecto aborda el problema como una **clasificación binaria**:
- `0` → máquina funcionando correctamente  
- `1` → falla de la máquina

---

## 🎯 Objetivo de Machine Learning

- **Tipo de problema:** Clasificación binaria
- **Variable objetivo:** `fail`
- **Resultado esperado:** modelo capaz de predecir fallas a partir de lecturas de sensores

---

## 📊 Dataset

El conjunto de datos contiene registros de sensores de distintas máquinas:

### Variables principales
- `footfall`: cantidad de personas/objetos cerca de la máquina
- `tempMode`: modo de temperatura
- `AQ`: índice de calidad del aire
- `USS`: sensor ultrasónico (proximidad)
- `CS`: consumo de corriente eléctrica
- `VOC`: compuestos orgánicos volátiles
- `RP`: rotación / RPM
- `IP`: presión de entrada
- `Temperature`: temperatura operativa
- `fail`: indicador de falla (target)

---

## 🧪 Metodología

### 1. Análisis exploratorio de datos (EDA)
- Revisión de valores nulos y duplicados
- Análisis de distribuciones
- Matriz de correlación para selección de variables

### 2. Selección de características
- Análisis de correlación
- Identificación de variables con mayor relación con la falla (`AQ`, `VOC`)

### 3. Preparación de datos
- Eliminación de duplicados
- Escalado de variables con `StandardScaler`
- División train / test respetando proporciones

---

## 🤖 Modelos entrenados

Se entrenaron y compararon múltiples enfoques:

### Deep Learning
- Red neuronal feed-forward con TensorFlow/Keras
- Activaciones ReLU y salida sigmoide
- Early stopping para evitar overfitting

### Machine Learning clásico
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- Decision Tree
- XGBoost Classifier

---

## ⚙️ Optimización de modelos

- Búsqueda de hiperparámetros con `GridSearchCV`
- Evaluación mediante:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Matriz de confusión

---

## 📈 Resultados

- Los modelos basados en **ensembles** (Random Forest, Gradient Boosting, XGBoost) mostraron el mejor desempeño
- El uso de escalado mejoró la estabilidad de los modelos
- La comparación entre múltiples algoritmos permitió identificar el enfoque más robusto para este problema

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **pandas, numpy**
- **matplotlib, seaborn**
- **scikit-learn**
- **TensorFlow / Keras**
- **XGBoost**

---

## 📂 Estructura del repositorio

├── datamaq.csv
├── Predicción de fallas de máquinas usando datos de sensores.py
├── README.md


---

## 🚀 Próximos pasos

- Manejo de desbalance de clases (SMOTE / class weights)
- Evaluación con métricas orientadas a negocio (Recall de fallas)
- Feature importance y explainability (SHAP)
- Detección temprana de anomalías
- Deploy del modelo para monitoreo en tiempo real

---

## 👤 Autor

**Flavia Hepp**  
Data Scientist en formación  
