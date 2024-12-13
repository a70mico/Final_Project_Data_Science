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
import os
from PIL import Image

# Serialización del modelo --------------------------------------------
from joblib import dump

# Streamlit --------------------------------------------
import streamlit as st


# Configuración de la página
st.set_page_config(
    page_title="MKT-Bancario - ML",
    page_icon="📊",
    layout="wide"
    )

# Carátula principal
st.title("🚀 El Rescate de las Campañas Perdidas")
st.markdown("""
### Proyecto Final - Bootcamp Data Science
#### Autores: Alvaro Diaz y Rodrigo Pinedo
**4Geeks Academy**
""")

# Obtener la ruta absoluta del directorio donde está el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta absoluta del archivo de imagen
image_path = os.path.join(BASE_DIR, "../streamlit/caratula.png")

# Imagen principal
if os.path.exists(image_path):
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))  # Redimensionar a 256x256
    st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
else:
    st.error(f"No se encontró la imagen en la ruta: {image_path}")

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

# Sidebar para la navegación
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Inicio",
        "El Rescate de las Campañas Perdidas",
        "La misión del rescate",
        "Desafíos abordados",
        "Herramientas y metodologías",
        "Hallazgos Clave",
        "Análisis Exploratorio de Datos (EDA)",
        "Resultados",
        "Puesta en acción"
    ]
)

# Sección: Inicio
if menu == "Inicio":
    st.title("🚀 El Rescate de las Campañas Perdidas")
    st.markdown("""
    ## Proyecto Final - Bootcamp Data Science
    Bienvenidos al viaje para transformar decisiones bancarias basadas en datos.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    image_inicio = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(image_inicio)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 1: El Rescate de las Campañas Perdidas
elif menu == "El Rescate de las Campañas Perdidas":
    st.title("El Rescate de las Campañas Perdidas")
    st.markdown("""
    ### El Problema
    Las campañas tradicionales dependen de llamadas masivas con baja tasa de éxito:
    - Solo el 2-5% de las campañas tienen éxito.
    - Recursos desperdiciados.
    """)
    # Gráfico de barras
    data = pd.DataFrame({"Estrategia": ["Masiva", "Optimizada"], "Tasa de Éxito (%)": [5, 20]})
    fig = px.bar(data, x="Estrategia", y="Tasa de Éxito (%)", title="Comparativa de Estrategias")
    st.plotly_chart(fig)


# Sección 2: La Misión del Rescate
elif menu == "La misión del rescate":
    st.title("La Misión del Rescate")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_2_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_2_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")

# Sección 3: Desafíos abordados
elif menu == "Desafíos abordados":
    st.title("Desafíos abordados")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 4: Herramientas y metodologías
elif menu == "Herramientas y metodologías":
    st.title("Herramientas y metodologías")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 5: Hallazgos Clave
elif menu == "Hallazgos Clave":
    st.title("Hallazgos Clave")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 6: Análisis Exploratorio de Datos (EDA)
elif menu == "Análisis Exploratorio de Datos (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 7: Resultados
elif menu == "Resultados":
    st.title("Resultados")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


# Sección 8: Puesta en acción
elif menu == "Puesta en acción":
    st.title("Puesta en acción")
    st.markdown("""
    Nuestro objetivo es transformar campañas ineficientes en estrategias optimizadas usando Ciencia de Datos:
    1. Identificar patrones clave en los datos.
    2. Segmentar clientes según su probabilidad de aceptación.
    3. Maximizar la tasa de éxito y reducir el costo.
    """)
    # Obtener la ruta absoluta del directorio donde está el script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta del archivo de imagen
    img_3_0 = os.path.join(BASE_DIR, "../streamlit/inicio.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_3_0)
        img_resized = img.resize((256, 256))  # Redimensionar a 256x256
        st.image(img_resized, use_container_width=False, caption="Decisiones basadas en evidencias, impulsadas por datos.")
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")


