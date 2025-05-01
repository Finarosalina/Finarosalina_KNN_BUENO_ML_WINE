import ipyleaflet
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import pymysql
from dotenv import load_dotenv
import requests
from sklearn import datasets, model_selection, metrics
import seaborn as sns
import sqlalchemy
import sympy
import xgboost as xgb

total_data= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv", sep=';')

total_data.head()
# label es el target
total_data.columns
print(total_data.info())
total_data.describe()
total_data['quality'].value_counts().sort_index()


df = total_data.copy() 
# Opción 1
df['label_v1'] = df['quality'].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))

# Opción 2
df['label_v2'] = df['quality'].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2))

df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop(['quality', 'label_v1', 'label_v2'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_v1 = df['label_v1']
y_v2 = df['label_v2']


X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_scaled, y_v1, test_size=0.3, random_state=42, stratify=y_v1)
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_scaled, y_v2, test_size=0.3, random_state=42, stratify=y_v2)

#  Función para evaluar modelo KNN

def evaluate_knn(X_train, X_test, y_train, y_test, label_name):
    print(f"Evaluación para {label_name}")
    train_accuracies = []
    test_accuracies = []
    best_k = 1
    best_acc = 0
    best_model = None
    k_values = range(1, 21)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)

        if acc_test > best_acc:
            best_acc = acc_test
            best_k = k
            best_model = knn

    # Reporte
    print(f"\n Mejor k: {best_k} con Accuracy de Test: {best_acc:.4f}")
    print("\n Reporte de clasificación (Test):")
    print(classification_report(y_test, best_model.predict(X_test)))

    # Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_accuracies, label='Entrenamiento', marker='o')
    plt.plot(k_values, test_accuracies, label='Prueba', marker='s')
    plt.axvline(best_k, color='r', linestyle='--', label=f'Mejor k = {best_k}')
    plt.title(f'Accuracy vs k - {label_name}')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_k, best_model


print("\n🔍 Comparación entre esquemas de clasificación")
k1, model1 = evaluate_knn(X_train_v1, X_test_v1, y_train_v1, y_test_v1, "Esquema 1 (0:<=4, 1:5-6, 2:>=7)")
k2, model2 = evaluate_knn(X_train_v2, X_test_v2, y_train_v2, y_test_v2, "Esquema 2 (0:<=5, 1:6, 2:>=7)")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar los datos nuevamente
df = pd.read_csv(
    "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv", sep=';')

# (0: ≤5, 1: 6, 2: ≥7)
df['label'] = df['quality'].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2))

X = df.drop(['quality', 'label'], axis=1)
y = df['label']

# Normalización de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División estratificada para mantener la misma proporción de clases
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Búsqueda del mejor k (usando GridSearchCV)
param_grid = {'n_neighbors': range(13, 18)}
grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=5, scoring='f1_weighted', n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"\n Mejor valor de k encontrado: {best_k}")

# Entrenar el modelo final
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Evaluación detallada
def detailed_evaluation(model, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)

    print("\nDistribución de clases (test):")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())

    print("\nMatriz de confusión (test):")
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title("Matriz de Confusión")
    plt.show()

    print("\n📄 Reporte de clasificación (test):")
    print(classification_report(y_test, y_pred_test, digits=3))

    print(f" F1-score ponderado: {f1_score(y_test, y_pred_test, average='weighted'):.3f}")
    print(f" Accuracy: {accuracy_score(y_test, y_pred_test):.3f}")

# Ejecutar evaluación
print("EVALUACIÓN FINAL DEL MODELO OPTIMIZADO (ESQUEMA 2)")
detailed_evaluation(final_model, X_train, X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# 1. Undersampling para balancear las clases
undersampler = RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = undersampler.fit_resample(X_train, y_train)

# 2. Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)  # Escalamos el conjunto de entrenamiento
X_test_scaled = scaler.transform(X_test)  # Escalamos el conjunto de prueba con el mismo scaler

# 3. Modelo KNN ponderado
knn = KNeighborsClassifier(n_neighbors=13, weights='distance')
knn.fit(X_train_scaled, y_train_bal)

# 4. Evaluación del modelo
test_preds = knn.predict(X_test_scaled)

# 5. Resultados finales
print("Clasificación:")
print(classification_report(y_test, test_preds, digits=3))
print(f" Balanced Accuracy: {balanced_accuracy_score(y_test, test_preds):.3f}")
print(f"F1 w: {f1_score(y_test, test_preds, average='weighted'):.3f}")

import joblib

# Guardar el modelo entrenado
joblib.dump(final_model, '//workspaces/Finarosalina_KNN_BUENO_ML_WINE/models/final_model.pkl')

def predict_wine_quality(features):
    # Escalar las características
    features_scaled = scaler.transform([features])
    
    # Predecir la calidad utilizando el modelo KNN
    prediction = knn.predict(features_scaled)
    
    # Interpretar la predicción
    if prediction == 0:
        return "Este vino probablemente sea de baja calidad "
    elif prediction == 1:
        return "Este vino probablemente sea de calidad media "
    else:
        return "Este vino probablemente sea de alta calidad "

# Probar la función con un ejemplo
print(predict_wine_quality([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]))

def predict_wine_quality(features, model, scaler):
    # Escalar las características
    features_scaled = scaler.transform([features])
    
    # Predecir la calidad utilizando el modelo final
    prediction = model.predict(features_scaled)
    
    # Interpretar la predicción
    if prediction == 0:
        return "Este vino probablemente sea de baja calidad."
    elif prediction == 1:
        return "Este vino probablemente sea de calidad media."
    else:
        return "Este vino probablemente sea de alta calidad."

# Probar la función con un ejemplo
print(predict_wine_quality([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4], final_model, scaler))

import joblib

# Guardar el modelo en la ruta especificada
model_path = '/workspaces/Finarosalina_KNN_BUENO_ML_WINE/models/knn_model.pkl'
joblib.dump(knn, model_path)
print(f"Modelo guardado en: {model_path}")

import json

with open('/workspaces/Finarosalina_KNN_BUENO_ML_WINE/src/explore.ipynb', 'r') as f:
    notebook_content = json.load(f)


code_cells = [cell['source'] for cell in notebook_content['cells'] if cell['cell_type'] == 'code']


code = '\n'.join([''.join(cell) for cell in code_cells])


with open('/workspaces/Finarosalina_KNN_BUENO_ML_WINE/src/app.py', 'w') as f:
    f.write(code)