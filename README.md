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
