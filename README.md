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
