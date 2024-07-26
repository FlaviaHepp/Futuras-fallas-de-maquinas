"""
Descripción general del conjunto de datos
Este conjunto de datos contiene datos de sensores recopilados de varias máquinas con el objetivo de predecir fallas de las máquinas con 
anticipación. Incluye una variedad de lecturas de sensores, así como las fallas de las máquinas registradas.

Columnas Descripción
footfall: La cantidad de personas u objetos que pasan por la máquina.
tempMode: El modo o configuración de temperatura de la máquina.
AQ: Índice de calidad del aire cerca de la máquina.
USS: Datos del sensor ultrasónico, que indican mediciones de proximidad.
CS: Lecturas del sensor actual, que indican el uso de corriente eléctrica de la máquina.
VOC: Nivel de compuestos orgánicos volátiles detectado cerca de la máquina.
RP: Posición rotacional o RPM (revoluciones por minuto) de las partes de la máquina.
IP: Presión de entrada a la máquina.
Temperature: La temperatura de funcionamiento de la máquina.
fail: Indicador binario de falla de la máquina (1 para falla, 0 para sin falla)."""

#Importando bibliotecas necesarias
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('datamaq.csv')
print(data.head())

missing_values = data.isnull().sum
print(data.isnull().sum().to_markdown(numalign="left", stralign="left"))

#Crear matriz de correlación para la selección de funciones
#matriz de correlación

cor = data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(cor, annot=True, cmap = "cool")
plt.show()

#Según la matriz de correlación, la mejor característica es AQ y VOC
#Así que usaremos estas funciones para predecir el objetivo

final_data = data[['AQ', 'VOC', 'fail']]

final_data.head()

#train test divide y convierte los datos en una matriz numpy
# x tren y tren x prueba y prueba

x = final_data.drop('fail', axis=1).values

y = final_data['fail'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Construir las redes neuronales

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(4, activation='relu',input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrenar al modelo
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')

model.fit(x_train, y_train, epochs=300, callbacks=[early_stopping])

model.evaluate(x_test, y_test)

data.info()

print(data.isnull().sum())
print(data.duplicated().sum())

data.drop_duplicates(inplace=True)
print(data['fail'].value_counts())

X = data.drop(columns=['fail'])
y = data['fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

param_grid = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.02, 0.3],
    'max_depth': [1, 3, 5, 7, 9]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=0)

grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Mejores parámetros: {best_params}')
print(f'Mejor puntuación: {best_score:.2f}')

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Ada Boost': AdaBoostClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: {accuracy:.2f}')
    
gb = GradientBoostingClassifier()

gb_params = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.02, 0.3],
    'max_depth': [1, 3, 5, 7, 9]
}

grid_search = GridSearchCV(gb, param_grid=gb_params, cv=3, scoring='accuracy', verbose=0)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Mejores parámetros: {best_params}')
print(f'Mejor puntuación: {best_score:.2f}')

rf = RandomForestClassifier()

rf_params = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [1, 3, 5, 7, 9]
}

grid_search = GridSearchCV(rf, param_grid=rf_params, cv=3, scoring='accuracy', verbose=0)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Mejores parámetros: {best_params}')
print(f'Mejor puntuación: {best_score:.2f}')

ab = AdaBoostClassifier()

ab_params = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.02, 0.3],
}

grid_search = GridSearchCV(ab, param_grid=ab_params, cv=3, scoring='accuracy', verbose=0)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Mejores parámetros: {best_params}')
print(f'Mejor puntuación: {best_score:.2f}')

et = ExtraTreesClassifier()

et_params = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [1, 3, 5, 7, 9]
}

grid_search = GridSearchCV(et, param_grid=et_params, cv=3, scoring='accuracy', verbose=0)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Mejores parámetros: {best_params}')
print(f'Mejor puntuación: {best_score:.2f}')

# Divida los datos en características y variable de destino
X = data.drop('fail', axis=1)
y = data['fail']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de prueba de tren
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenamiento de modelos: clasificador de bosque aleatorio
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluación del modelo
y_pred = rf_classifier.predict(X_test)
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nInforme de clasificación:\n", classification_report(y_test, y_pred))
print("\nPuntuación de precisión:", accuracy_score(y_test, y_pred))

