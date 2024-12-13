# Step 0: Importar librerias y modelos
# Librerías principales -------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y transformaciones ----------------------------------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, 
    OneHotEncoder, 
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

# Modelos de Machine Learning ------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from xgboost import XGBClassifier

# Métricas de evaluación ----------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Búsqueda de hiperparámetros ------------------------------------------
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Visualización interactiva -------------------------------------------
import plotly.express as px

# Manejo de datos externos --------------------------------------------
import requests
from io import StringIO

# Serialización del modelo --------------------------------------------
from joblib import dump

# Streamlit --------------------------------------------
import streamlit as st


# Configuración de la página
st.set_page_config(
    page_title="Marketing Bancario - ML",
    page_icon="💡",
    layout="centered"
    )

# Carátula principal
st.title("🚀 La Revolución del Marketing Bancario")
st.markdown("""
### Proyecto Final - Bootcamp Data Science
**4Geeks Academy**
""")

# Imagen principal
st.image("streamlit/caratula.png", use_container_width=True, caption="Decisiones basadas en evidencias, impulsadas por datos.")

# Subtítulo motivacional
st.markdown("""
#### **¿Qué encontrarás en este proyecto?**
- Exploración de datos reales de campañas bancarias.
- Visualizaciones interactivas para entender patrones.
- Modelos de Machine Learning que optimizan decisiones estratégicas.
""")

# Call to Action
if st.button("Comienza tu viaje"):
    st.success("¡Navega por las secciones para descubrir más!")

# Sidebar para navegar
st.sidebar.title("Explora el Proyecto")
menu = st.sidebar.radio("Secciones:", ["Inicio", "Análisis de Datos", "Modelos Predictivos", "Conclusiones"])

if menu == "Inicio":
    st.write("Estás en la pantalla de inicio.")
elif menu == "Análisis de Datos":
    st.write("Aquí analizamos los datos...")
elif menu == "Modelos Predictivos":
    st.write("Esta sección muestra el desempeño de los modelos.")
elif menu == "Conclusiones":
    st.write("Resumen de hallazgos y próximos pasos.")
