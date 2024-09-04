import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Título principal de la aplicación
st.title("Análisis y Predicción del Dataset Titanic usando Árbol de Decisión Basado en Mitchell")

# Descripción del dataset y su origen
st.markdown("""
Esta aplicación utiliza un modelo de Árbol de Decisión basado en el enfoque de Tom Mitchell para predecir la supervivencia de pasajeros del Titanic.
El análisis se realiza usando el dataset de la competencia Titanic de Kaggle, disponible en:
[Dataset Titanic - Kaggle](https://www.kaggle.com/competitions/titanic/data?select=train.csv).
""")

# Cargar el archivo CSV de entrenamiento
train_data = pd.read_csv('train.csv')

# Previsualizar los datos de entrenamiento
st.write("### Vista previa de los datos de entrenamiento:")
st.write(train_data.head())

# Preprocesamiento: Convertir columnas categóricas a variables numéricas
def preprocess_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)  # Eliminar columnas irrelevantes, incluyendo PassengerId
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Codificar sexo
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Codificar embarque
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Llenar valores faltantes en Age con la mediana
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Llenar valores faltantes en Embarked con la moda
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Llenar valores faltantes en Fare con la mediana
    return df

# Preprocesar los datos de entrenamiento
train_data = preprocess_data(train_data)

# Separar las características (X) y la variable objetivo (y)
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# División de los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de árbol de decisión usando el criterio 'entropy' para aplicar ganancia de información
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión detallada
st.write("### Matriz de Confusión:")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
ax.set_title('Matriz de Confusión')
st.pyplot(fig)

# Mostrar el reporte de clasificación
st.write("### Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Interfaz de usuario para ingresar nuevos datos
st.sidebar.header("Ingresar nuevos datos para la predicción")

def user_input_features():
    pclass = st.sidebar.selectbox('Clase de Pasajero (Pclass)', (1, 2, 3))
    sex = st.sidebar.selectbox('Sexo (Sex)', ('male', 'female'))
    age = st.sidebar.slider('Edad (Age)', 0, 80, 30)
    sibsp = st.sidebar.number_input('Número de hermanos/esposos abordo (SibSp)', 0, 10, 0)
    parch = st.sidebar.number_input('Número de padres/hijos abordo (Parch)', 0, 10, 0)
    fare = st.sidebar.number_input('Tarifa del pasaje (Fare)', 0.0, 600.0, 32.2)
    embarked = st.sidebar.selectbox('Puerto de Embarque (Embarked)', ('C', 'Q', 'S'))

    data = {
        'Pclass': pclass,
        'Sex': 0 if sex == 'male' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': 0 if embarked == 'C' else (1 if embarked == 'Q' else 2)
    }
    return pd.DataFrame([data])

# Obtener datos de entrada del usuario
input_df = user_input_features()

# Generar predicción con los nuevos datos
if st.sidebar.button("Predecir"):
    prediction = model.predict(input_df)
    st.write("## Resultado de la Predicción")
    st.write(f"El resultado de la predicción es: {prediction[0]}")

    # Interpretación del resultado
    st.write("### Interpretación del Resultado")
    st.write("""
    El resultado es 0 (No sobrevivió) o 1 (Sobrevivió). Basado en las características ingresadas, 
    el modelo de árbol de decisión ha determinado si el pasajero probablemente habría sobrevivido 
    (1) o no (0). Esta predicción está basada en los patrones encontrados en los datos históricos 
    de pasajeros del Titanic.
    """)
