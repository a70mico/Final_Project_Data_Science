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

# Visualización interactiva --------------------------------------------
import plotly.express as px

# Manejo de datos externos ---------------------------------------------
import requests
from io import StringIO
import os
from PIL import Image

# Serialización del modelo ---------------------------------------------
from joblib import dump
from joblib import load

# Streamlit ------------------------------------------------------------
import streamlit as st

# Cargar df_raw

# Obtener la ruta absoluta del directorio donde está el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta absoluta al archivo CSV
data_path = os.path.join(BASE_DIR, "../data/raw_2/bank-full.csv")

# Intentar cargar el archivo CSV
try:
    df_raw = pd.read_csv(data_path, sep=';')
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {data_path}")
except Exception as e:
    print(f"Ocurrió un error al cargar el archivo: {e}")

# Cargar df_clean
# Construir la ruta absoluta al archivo CSV
data_clean_path = os.path.join(BASE_DIR, "../data/processed/df_clean_bank.csv")

# Intentar cargar el archivo CSV
try:
    df_clean = pd.read_csv(data_clean_path, index_col=0)
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {data_clean_path}")
except Exception as e:
    print(f"Ocurrió un error al cargar el archivo: {e}")

# EDA ------------------------------------------------------------------
df = df_clean.copy()
# Crear data set de entrenamiento y test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)



# Configuración de la página -------------------------------------------
st.set_page_config(
    page_title="MKT-Bancario - ML",
    page_icon="📊",
    layout="wide"
    )

# Carátula principal
st.title("🚀 El Rescate de las Campañas Perdidas")

# Menú desplegable para carátula
with st.expander("Presentación"):
    st.markdown("""
        ### Proyecto Final - Bootcamp Data Science
        #### Autores: Alejandro Diaz y Rodrigo Pinedo
        **4Geeks Academy**
    """)

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
        "1. El Rescate de las Campañas Perdidas",
        "2. La misión del rescate",
        "3. Desafíos abordados",
        "4. Herramientas y metodologías",
        "5. Hallazgos Clave",
        "6. Análisis Exploratorio de Datos (EDA)",
        "7. Resultados",
        "8. Puesta en acción",
        "9. Predicción"
    ]
)

# Sección: Inicio ------------------------------------------------------
if menu == "Inicio":
    st.title(" Inicio")

    # Construir la ruta absoluta del archivo de imagen
    image_inicio = os.path.join(BASE_DIR, "../streamlit/inicio.png")
    
    # Crear las tres columnas
    col1, col2, col3 = st.columns([1, 1, 1])

    # Contenido de la columna izquierda
    with col1:
        st.markdown("#### Anteriormente")
        st.markdown("- Decisiones basadas en suposiciones")
        st.markdown("- Incertidumbre elevada")
        st.markdown("- Rendimientos ineficientes")
        st.markdown("")
        st.markdown("#### Limitaciones de anteriores")
        st.markdown("- Parece complicado")
        st.markdown("- Desconocimiento")

    # Contenido de la columna central (imagen)
    with col2:
        if os.path.exists(image_inicio):
            img = Image.open(image_inicio)
            img_resized = img.resize((256, 256))  # Redimensionar a 256x256
            st.image(img_resized, use_container_width=False)
        else:
            st.error(f"No se encontró la imagen en la ruta: {image_inicio}")
        st.markdown("#### Apoyo de herramientas tecnológicas")
        st.markdown("Decisiones basadas en evidencias, impulsadas por datos.")

    # Contenido de la columna derecha
    with col3:
        st.markdown("#### Apoyo de la Ciencia de Datos")
        st.markdown("- Decisiones basadas en evidencias")
        st.markdown("- Mayor ventaja competitiva")
        st.markdown("")
        st.markdown("")
        st.markdown("#### Ventajas")
        st.markdown("- Enfrentar el futuro con confianza")
        st.markdown("- Cambios estructurados")

    # Mensaje de bienvenida adicional
    st.markdown("#### Haz clic en el menú lateral para explorar las secciones.")


# Sección 1: 1. El Rescate de las Campañas Perdidas -----------------------
elif menu == "1. El Rescate de las Campañas Perdidas":
    st.title("1. El Rescate de las Campañas Perdidas")
    st.markdown("""
    ### El Problema
    *"El banco enfrenta el reto de mejorar el desempeño de sus campañas 
    de marketing telefónico, que actualmente tienen una baja tasa de éxito."*
    """)
    # Construir la ruta absoluta del archivo de imagen
    image_s_1 = os.path.join(BASE_DIR, "../streamlit/s_1.png")
    
    # Crear las tres columnas
    col1, col2 = st.columns([1, 1])

    # Contenido de la columna izquierda
    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

        # Imagen gerente
        if os.path.exists(image_s_1):
            img = Image.open(image_s_1)
            img_resized = img.resize((256, 256))  # Redimensionar a 256x256
            st.image(img_resized, use_container_width=False)
        else:
            st.error(f"No se encontró la imagen en la ruta: {image_s_1}")

        st.markdown("""
                    - Solo el 11.7% de las campañas tienen éxito.
                    - Se evidencia que el dataset se encuentra desbalanceado.
        """)

    # Contenido de la columna central (imagen)
    with col2:
        # Conteo de valores de la variable objetivo
        target_counts = df_raw['y'].value_counts().reset_index()
        target_counts.columns = ['Target', 'Count']

        # Crear el gráfico interactivo con Plotly
        fig = px.pie(
            target_counts, 
            names='Target', 
            values='Count', 
            title='Distribución de la Variable Objetivo',
            color_discrete_map={'no': '#FF6F61', 'yes': '#6A89CC'}  # Mapear colores
        )

        # Ajustar tamaño de las fuentes
        fig.update_traces(textinfo='percent+label', textfont_size=12)

        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")

# Sección 2: 2. La Misión del Rescate -------------------------------------
elif menu == "2. La misión del rescate":
    st.title("2. La Misión del Rescate")

    # Construir la ruta absoluta del archivo de imagen
    img_2_0 = os.path.join(BASE_DIR, "../streamlit/s_2.png")

    # Construir columnas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("### **Objetivo:**")
        st.markdown("""
        Predecir cuando un cliente del banco realizará un depósito a plazo.

        A través de identificar patrones en los datos históricos para 
        optimizar las campañas y mejorar la tasa de éxito.
    """)
        
    with col2:

        # Imagen principal
        if os.path.exists(image_path):
            img = Image.open(img_2_0)
            img_resized = img.resize((256, 256))  # Redimensionar a 256x256
            st.image(img_resized, use_container_width=False)
        else:
            st.error(f"No se encontró la imagen en la ruta: {image_path}")
    
    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")

# Sección 3: 3. Desafíos abordados ----------------------------------------
elif menu == "3. Desafíos abordados":
    st.title("3. Desafíos abordados")
    st.markdown("""
    El análisis afrontó desafíos interesantes para procesar los datos, permitiendo
    mejorar el poder predictivo de las características.
    """)
        # Construir columnas
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Crear tabla con las variables y su tipo de dato
        data_types = pd.DataFrame({
            'Tipo de Dato': df_raw.dtypes.astype(str)  # Convertir a string para visualización
        })

        # Mostrar la tabla en Streamlit
        st.write(data_types)
        
        
    with col2:
        st.markdown("#### Info dataset original")
        st.markdown("""
            - Dataset Original:
            - Registros: 45,211
            - Variables: 16 características
            - Meta: 1 objetivo a predecir (y)
        """)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### Info dataset limpio")
        st.markdown("""
            - Dataset Original:
            - Registros: 44,724
            - Variables: 15 características
            - Meta: 1 objetivo a predecir (y)
        """)

    with col3:
        # Crear tabla df_clean
        data_types2 = pd.DataFrame({
            'Tipo de Dato': df_clean.dtypes.astype(str)  # Convertir a string para visualización
        })

        # Mostrar la tabla en Streamlit
        st.write(data_types2)

    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")


# Sección 4: 4. Herramientas y metodologías -------------------------------
elif menu == "4. Herramientas y metodologías":
    st.title("4. Herramientas y metodologías")
    st.markdown("Todo lo utilizado para el proyecto se describe a continuación:")
    # Crear columnas
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:    
        st.markdown("""
            **Herramientas:**
            - Pandas
            - Numpy
            - Seaborn
            - Matplotlib
            - Sklearn
            - Joblib
            - Python
            - Jupyter
            - Streamlit
        """)

    with col2:
        # Construir la ruta absoluta del archivo de imagen
        img_4_0 = os.path.join(BASE_DIR, "../streamlit/s_4.png")

        # Imagen principal
        if os.path.exists(image_path):
            img = Image.open(img_4_0)
            img_resized = img.resize((256, 256))  # Redimensionar a 256x256
            st.image(img_resized, use_container_width=False)
        else:
            st.error(f"No se encontró la imagen en la ruta: {image_path}")

    with col3:    
        st.markdown("""
            **Metodologias:**
            - Estadística descriptiva e inferencial
            - Análisis exploratorio de datos
            - Transformación de datos (log, Yeo-Johnson, clasificación y binarias)
            - Encodear características
            - Modelos de Machine Learning (Random Forest, XGBoost y LGBM)
            - Mejoramiento de hiperparametros de los modelos ML
            - Técnicas de evaluación de modelos ML
        """)

    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")


# Sección 5: 5. Hallazgos Clave -------------------------------------------
elif menu == "5. Hallazgos Clave":
    st.title("5. Hallazgos Clave")
    st.markdown("""
    En este apartado explicaremos las características que tuvieron comportamientos a considerarse:  
    """)

    # Crear columnas Var_1
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### **Age**")
        st.markdown(""""
            Los datos mayores a 70 años son outliers que pueden 
            sesgar el análisis, por lo que los eliminamos para 
            garantizar un modelo más robusto.
        """)

    with col2:
        # Crear el boxplot usando Plotly Express
        fig = px.box(
            df_raw, 
            x='age', 
            title="Distribución de Edad de los Clientes",
            labels={'age': 'Edad'},  # Etiquetas personalizadas
            template="plotly_white",  # Tema visual limpio
            color_discrete_sequence=["#636EFA"]  # Color del boxplot
        )

        # Añadir anotaciones narrativas
        fig.update_layout(
            xaxis=dict(
                title="Edad de los Clientes",
                title_standoff=20  # Separación del título del eje X
            ),
            yaxis_title="",  # Eliminar etiquetas del eje Y
            annotations=[
                dict(
                    x=0.5, y=-0.3, xref="paper", yref="paper", showarrow=False,
                    text="Los outliers representan edades que se desvían significativamente del rango típico."
                )
            ],
            height=400,  # Ajustar altura del gráfico
            title_x=0  # Centrar el título
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # Crear columnas Var_2
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:

        # Gráfico original
        fig_original = px.histogram(
            df_raw,
            x='balance',
            nbins=30,
            title="Distribución Original",
            labels={'balance': 'Balance'},
            color_discrete_sequence=['blue'],
            template='plotly_white'
        )
        fig_original.update_layout(title_x=0)

        # Gráfico transformado con Yeo-Johnson
        fig_transformed = px.histogram(
            df_clean,
            x='balance_yeojohnson',
            nbins=30,
            title="Distribución Transformada (Yeo-Johnson)",
            labels={'balance_yeojohnson': 'Balance Transformado (Yeo-Johnson)'},
            color_discrete_sequence=['orange'],
            template='plotly_white'
        )
        fig_transformed.update_layout(title_x=0)

        # Mostrar ambos gráficos en Streamlit        
        st.plotly_chart(fig_original, use_container_width=True)

    with col2:
        st.plotly_chart(fig_transformed, use_container_width=True)


    with col3:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Balance**")
        st.markdown("""
            Presentaba una distribución sesgada con outliers
            extremos, lo que dificultaba el modelado. Aplicamos 
            la transformación Yeo-Johnson para normalizar los datos.
        """)

    # Crear columnas Var_3
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Campaign**")
        st.markdown("""
            Las campañas tenían una distribución altamente sesgada. 
            La transformación logarítmica permitió comprimir la 
            escala y mejorar la estabilidad del modelo.
        """)
    with col2:

        # Gráfico Original
        fig_original = px.histogram(
            df_raw,
            x='campaign',
            nbins=30,
            title="Distribución Original de Campaign",
            labels={'campaign': 'Número de Campañas'},
            color_discrete_sequence=['blue'],
            template='plotly_white'
        )
        fig_original.update_layout(title_x=0)

        # Gráfico Transformado (Log-Transform)
        fig_log_transform = px.histogram(
            df_clean,
            x='campaign_log',
            nbins=30,
            title="Distribución Transformada (Log-Transform)",
            labels={'campaign_log': 'Log Transform de Campañas'},
            color_discrete_sequence=['green'],
            template='plotly_white'
        )
        fig_log_transform.update_layout(title_x=0)

        # Mostrar gráficos lado a lado en Streamlit
        st.plotly_chart(fig_original, use_container_width=True)

    with col3:
            st.plotly_chart(fig_log_transform, use_container_width=True)


    # Crear columnas Var_4
    col1, col2 = st.columns([2, 1])

    with col1:
        # Crear el histograma usando Plotly Express
        fig = px.histogram(
            df_clean, 
            x='quarter', 
            title="Distribución de Trimestres",
            labels={'quarter': 'Trimestre'},  # Etiqueta personalizada para el eje X
            color_discrete_sequence=['#636EFA'],  # Color del gráfico
            template='plotly_white',  # Tema visual limpio
            text_auto=True  # Mostrar conteos encima de las barras
        )

        # Personalizar el diseño
        fig.update_layout(
            title_x=0,  # Título alineado a la izquierda
            xaxis_title="Trimestre",  # Etiqueta del eje X
            yaxis_title="Conteo",  # Etiqueta del eje Y
            bargap=0.2  # Espacio entre barras
        )

        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Month**")
        st.markdown("""
            Agrupamos los meses en trimestres para simplificar el 
            análisis y capturar estacionalidad en las campañas.
        """)

    # Crear columnas Var_5
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Pdays**")
        st.markdown("""
            Convertimos pdays en una variable binaria (contactado/no 
            contactado) para simplificar el análisis y mejorar la 
            interpretabilidad.
        """)
    with col2:
        # Preparar los datos para el gráfico
        contact_counts = df_clean['pdays_tran'].value_counts().reset_index()
        contact_counts.columns = ['Contactado', 'Frecuencia']

        # Crear el gráfico de barras usando Plotly Express
        fig = px.bar(
            contact_counts,
            x='Contactado',
            y='Frecuencia',
            title='Distribución de Contactados y No Contactados',
            labels={'Contactado': 'Contactado (1) o No Contactado (0)', 'Frecuencia': 'Frecuencia'},  # Etiquetas personalizadas
            template='plotly_white',  # Tema visual limpio
            color_discrete_sequence=['blue', 'red'] 
        )

        # Personalizar el diseño
        fig.update_layout(
            title_x=0,  # Título alineado a la izquierda
            xaxis=dict(
                tickmode='array', 
                tickvals=[0, 1], 
                ticktext=['No Contactado (0)', 'Contactado (1)']  # Etiquetas personalizadas para el eje X
            ),
            bargap=0.2  # Espacio entre las barras
        )

        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)


# Sección 6: 6. Análisis Exploratorio de Datos (EDA)-----------------------
elif menu == "6. Análisis Exploratorio de Datos (EDA)":
    st.title("6. Análisis Exploratorio de Datos (EDA)")
    st.markdown("""
        Es fundamental realizar este análisis de nuestros datos    
    """)
    st.markdown("")
    st.markdown("### **Info**")
    # Obtener la información de las columnas
    info_data = {
        "Columna": df_train.columns,
        "No. Valores No Nulos": df_train.notnull().sum().values,
        "Tipo de Dato": df_train.dtypes.values
    }

    # Crear el DataFrame
    info_df = pd.DataFrame(info_data)

    # Mostrar la tabla en Streamlit
    st.dataframe(info_df)

    st.markdown("")
    st.markdown("### **Describe**")
    
    # Mostrar estadísticas de variables numéricas
    numeric_stats = df_train.describe(include='number').T  # Transponer para formato tabular
    st.markdown("### Estadísticas Descriptivas de Variables Numéricas")
    st.dataframe(numeric_stats)

    # Mostrar estadísticas de variables categóricas
    category_stats = df_train.describe(include='object').T  # Transponer para formato tabular
    st.markdown("### Estadísticas Descriptivas de Variables Categóricas")
    st.dataframe(category_stats)

    st.markdown("")
    st.markdown("### **Análisis Univariado** *(numéricas)*")
    # Construir la ruta absoluta del archivo de imagen
    img_6_0 = os.path.join(BASE_DIR, "../streamlit/s_6_0.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_6_0)
        img_resized = img.resize((1000, 1000))  # Redimensionar a 1000x1000
        st.image(img_resized, use_container_width=False)
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")

    st.markdown("")
    st.markdown("### **Análisis Univariado** *(categoricas)*")
    # Construir la ruta absoluta del archivo de imagen
    img_6_1 = os.path.join(BASE_DIR, "../streamlit/s_6_1.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_6_1)
        img_resized = img.resize((1000, 1000))  # Redimensionar a 1000x1000
        st.image(img_resized, use_container_width=False)
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")

    st.markdown("")
    st.markdown("### **Análisis Bivariado** *(numéricas)*")
    # Construir la ruta absoluta del archivo de imagen
    img_6_2 = os.path.join(BASE_DIR, "../streamlit/s_6_2.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_6_2)
        img_resized = img.resize((1000, 1000))  # Redimensionar a 1000x1000
        st.image(img_resized, use_container_width=False)
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")

    st.markdown("")
    st.markdown("### **Análisis Bivariado** *(categoricas)*")
    # Construir la ruta absoluta del archivo de imagen
    img_6_3 = os.path.join(BASE_DIR, "../streamlit/s_6_3.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_6_3)
        img_resized = img.resize((1000, 1000))  # Redimensionar a 1000x1000
        st.image(img_resized, use_container_width=False)
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")

    st.markdown("")
    st.markdown("### **Correlación**")
    # Construir la ruta absoluta del archivo de imagen
    img_6_4 = os.path.join(BASE_DIR, "../streamlit/s_6_4.png")

    # Imagen principal
    if os.path.exists(image_path):
        img = Image.open(img_6_4)
        img_resized = img.resize((500, 500))  # Redimensionar a 5s00x500
        st.image(img_resized, use_container_width=False)
    else:
        st.error(f"No se encontró la imagen en la ruta: {image_path}")

    st.markdown("### Haz clic en el menú lateral para explorar las secciones.")

    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")


# Sección 7: 7. Resultados ------------------------------------------------
elif menu == "7. Resultados":
    st.title("7. Resultados")
    st.markdown("""
        Entrenamos tres modelos de ML para elegir el que mejor rendimiento tiene:
        - Random Forest
        - XGBoost
        - LGBM
    """)

    # Crear columnas
    col1, col2 = st.columns([2, 1])    

    with col1:
        # Datos de la tabla
        data = {
            "Model": ["Random Forest", "XGBoost", "LightGBM"],
            "Accuracy": [0.840805, 0.838122, 0.833762],
            "F1-Score (Class 1)": [0.543297, 0.544368, 0.538055],
            "Recall (Class 1)": [0.841948, 0.859841, 0.860835],
        }

        df_eval = pd.DataFrame(data)

        # Crear la tabla con Plotly
        import plotly.graph_objects as go
        table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_eval.columns),
                fill_color=["#d9ead3", "#fce5cd", "#cfe2f3", "#d9d2e9"],
                align="center",
                font=dict(size=14, color="black"),
            ),
            cells=dict(
                values=[df_eval[col] for col in df_eval.columns],
                fill_color="white",
                align="center",
                font=dict(size=12, color="black"),
                height=30  # Altura de las celdas
            ),
        )])

        # Ajustar altura y diseño general
        table_fig.update_layout(
            height=400,  # Altura total del gráfico
            margin=dict(l=0, r=0, t=10, b=10),  # Reducir márgenes para compactar
        )

        # Mostrar la tabla en Streamlit
        st.plotly_chart(table_fig)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("**Evaluación de los modelos de ML:**")
        st.markdown("""
            La métrica para determinar el mejor modelo a aplicar,
            es *Recall*. Debido a que nuestro dataset está desbalanceado.
        """)

    # Crear columnas
    col1, col2 = st.columns([1, 2])

    with col1:
        # Mostrar texto debajo de la gráfica
        st.markdown("""
            El modelo de clasificación predice con 86% de precisión si
            un cliente aceptará hacer el depósito a plazo fijo.
        """)
    
    with col2:
        # Crear el gráfico de líneas con Plotly
        df_melted = df_eval.melt(
            id_vars=["Model"], 
            value_vars=["Accuracy", "F1-Score (Class 1)", "Recall (Class 1)"],
            var_name="Metric", 
            value_name="Score"
        )
        line_fig = px.line(
            df_melted,
            x="Model", y="Score", color="Metric",
            title="Comparación de Modelos de ML",
            markers=True,
        )

        # Personalizar el diseño del gráfico
        line_fig.update_layout(
            title=dict(font=dict(size=18, family="Arial", color="black")),
            xaxis_title="Modelo",
            yaxis_title="Puntaje",
            legend_title="Métricas",
            margin=dict(l=0, r=0, t=50, b=10),  # Ajustar márgenes
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(line_fig, use_container_width=True)


# Sección 8: 8. Puesta en acción ------------------------------------------
elif menu == "8. Puesta en acción":
    st.title("8. Puesta en acción")
    
    # Crear columnas
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("""
        - El equipo de marketing podrá enfocar sus esfuerzos en clientes identificados como potenciales.
        - Gracias al proyecto el banco podrá utilizar sus recursos de marketing de manera eficiente.
        - El banco podrá tomar mejor decisiones con mayor confianza.
        """)

    with col2:
        # Construir la ruta absoluta del archivo de imagen
        img_8_0 = os.path.join(BASE_DIR, "../streamlit/s_8.png")

        # Imagen principal
        if os.path.exists(image_path):
            img = Image.open(img_8_0)
            img_resized = img.resize((256, 256))  # Redimensionar a 256x256
            st.image(img_resized, use_container_width=False)
        else:
            st.error(f"No se encontró la imagen en la ruta: {image_path}")
    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")


# Sección 9: 9. Predicción ------------------------------------------
elif menu == "9. Predicción":
    st.title("9. Predicción")

    # Ruta del modelo guardado
    model_path = os.path.join(BASE_DIR, "../models/bank_marketing_lgbm_model.joblib")

    # Cargar el modelo
    try:
        model = load(model_path)
        st.success("Modelo cargado exitosamente.")
    except FileNotFoundError:
        st.error("No se encontró el modelo guardado en la ruta especificada.")

    # Crear un formulario para recolectar datos del usuario
    st.markdown("### Introduce los datos del cliente:")
    with st.form("prediction_form"):
        age = st.number_input("Edad", min_value=18, max_value=95, step=1, value=35)
        job = st.selectbox("Trabajo", ["blue-collar", "admin.", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknow"])
        marital = st.selectbox("Estado civil", ["married", "single", "divorced"])
        education = st.selectbox("Educación?", ["primary", "secondary", "tertiary", "unknow"])
        default = st.selectbox("¿Tiene crédito en mora?", ["yes", "no"])
        housing = st.selectbox("¿Tiene hipoteca?", ["yes", "no"])
        loan = st.selectbox("¿Tiene préstamo personal?", ["yes", "no"])
        contact = st.selectbox("¿Tipo de contacto?", ["celullar", "telephone", "unknow"])
        day = st.number_input("¿Qué día lo contactaron?", step=1, value=1)
        duration = st.number_input("¿Tiempo de la llamada (segundos)?", step=0, value=3600)
        poutcome = st.selectbox("Resultado de la campaña previa", ["success", "failure", "other", "unknown"])
        balance = st.number_input("Balance", min_value=-1800, max_value=3000, step=1, value=0)
        campaign = st.number_input("Número de contactos durante la campaña", min_value=0, max_value=60, step=1, value=0)
        quarter = st.selectbox("Trimestre contactado", ["Q1", "Q2", "Q3", "Q4"])
        pdays = st.selectbox("¿Se lo contacto antes?", [0, 1])
        
        
        # Botón para realizar la predicción
        submitted = st.form_submit_button("Hacer Predicción")

    def preprocess_input_data(data):
        processed_data = pd.DataFrame()

        # Codificar 'job' (One-Hot Encoding manual)
        job_categories = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                        'retired', 'self-employed', 'services', 'student', 'technician', 
                        'unemployed', 'unknown']
        for job in job_categories:
            processed_data[f'cat__job_{job}'] = (data['job'] == job).astype(int)

        # Codificar 'marital'
        marital_categories = ['divorced', 'married', 'single']
        for marital in marital_categories:
            processed_data[f'cat__marital_{marital}'] = (data['marital'] == marital).astype(int)

        # Codificar 'education'
        education_categories = ['primary', 'secondary', 'tertiary', 'unknown']
        for education in education_categories:
            processed_data[f'cat__education_{education}'] = (data['education'] == education).astype(int)

        # Codificar 'default'
        processed_data['cat__default_no'] = (data['default'] == 'no').astype(int)
        processed_data['cat__default_yes'] = (data['default'] == 'yes').astype(int)

        # Codificar 'housing'
        processed_data['cat__housing_no'] = (data['housing'] == 'no').astype(int)
        processed_data['cat__housing_yes'] = (data['housing'] == 'yes').astype(int)

        # Codificar 'loan'
        processed_data['cat__loan_no'] = (data['loan'] == 'no').astype(int)
        processed_data['cat__loan_yes'] = (data['loan'] == 'yes').astype(int)

        # Codificar 'contact'
        contact_categories = ['cellular', 'telephone', 'unknown']
        for contact in contact_categories:
            processed_data[f'cat__contact_{contact}'] = (data['contact'] == contact).astype(int)

        # Codificar 'poutcome'
        poutcome_categories = ['failure', 'other', 'success', 'unknown']
        for poutcome in poutcome_categories:
            processed_data[f'cat__poutcome_{poutcome}'] = (data['poutcome'] == poutcome).astype(int)

        # Codificar 'quarter'
        quarter_categories = ['Q1', 'Q2', 'Q3', 'Q4']
        for quarter in quarter_categories:
            processed_data[f'cat__quarter_{quarter}'] = (data['quarter'] == quarter).astype(int)

        # Variables numéricas restantes
        processed_data['remainder__age'] = data['age']
        processed_data['remainder__day'] = data['day']
        processed_data['remainder__duration'] = data['duration']
        processed_data['remainder__balance_yeojohnson'] = data['balance_yeojohnson']
        processed_data['remainder__campaign_log'] = np.log1p(data['campaign_log'])
        processed_data['remainder__pdays_tran'] = data['pdays_tran']

        # Codificar 'pdays'
        # processed_data['remainder__pdays_tran'] = data['pdays_tran'].apply(lambda x: 0 if x == 'no' else 1).astype(int)

        return processed_data


    if submitted:
        # Crear un DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "day": [day],
            "duration": [duration],
            "poutcome": [poutcome],
            "balance_yeojohnson": [balance],
            "campaign_log": [campaign],
            "quarter": [quarter],
            "pdays_tran": [pdays]
        })

        # Procesar los datos
        processed_data = preprocess_input_data(input_data)

        # Verificar si las características coinciden
        expected_features = model.booster_.feature_name()  # Obtiene las características esperadas por el modelo
        missing_features = [col for col in expected_features if col not in processed_data.columns]

        # Agregar columnas faltantes con valor 0
        for col in missing_features:
            processed_data[col] = 0

        # Reordenar las columnas para que coincidan con el modelo
        processed_data = processed_data[expected_features]

        # Verificar si el modelo está cargado
        if 'model' in locals():
            # Hacer predicción
            prediction = model.predict(processed_data)
            prediction_prob = model.predict_proba(processed_data)

            # Mostrar el resultado
            if prediction[0] == 1:
                st.success(f"El modelo predice que el cliente **REALIZARA EL DEPOSITO A PLAZO** la oferta.")
            else:
                st.info(f"El modelo predice que el cliente **NO REALIZARA EL DEPOSITO A PLAZO** la oferta.")
            
            st.markdown(f"### Probabilidades:")
            st.write(f"- No Aceptará: {prediction_prob[0][0]:.2f}")
            st.write(f"- Aceptará: {prediction_prob[0][1]:.2f}")
        else:
            st.error("El modelo no está cargado. Verifica el archivo del modelo.")

    
    st.markdown("**Nota:** *Haz clic en el menú lateral para explorar las secciones.*")
