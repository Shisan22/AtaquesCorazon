# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Configuracion de la pagina
st.set_page_config(
    page_title="Prediccion de Ataque al Corazon",
    layout="centered"
)

# 2. Cargar el modelo y herramientas de preprocesamiento
@st.cache_resource
def cargar_modelo():
    filename = 'modelo-class.pkl'
    with open(filename, 'rb') as file:
        modelo, labelencoder, variables, min_max_scaler = pickle.load(file)
    return modelo, labelencoder, variables, min_max_scaler

modelo, labelencoder, variables, min_max_scaler = cargar_modelo()

# 3. Diseno de la Interfaz (UI)
st.title('Prediccion de Riesgo de Ataque al Corazon')
st.markdown("""
Bienvenido a la herramienta de diagnostico preventivo. Ingresa los indicadores medicos y demograficos 
del paciente en el formulario inferior para evaluar su nivel de riesgo utilizando nuestro modelo de Machine Learning.
""")

st.header("Datos del Paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Edad (anos)', min_value=1, max_value=100, value=40, step=1)
    hypertension = st.selectbox('¿Padece de Hipertension?', ['No', 'Yes'])
    heart_disease = st.selectbox('¿Tiene Enfermedad Cardiaca?', ['No', 'Yes'])

with col2:
    avg_glucose_level = st.number_input('Nivel Promedio de Glucosa', min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    ever_married = st.selectbox('¿Alguna vez se ha casado?', ['No', 'Yes'])
    # Se corrigieron las comillas internas para que coincidan con la data real
    smoking_status = st.selectbox('Estado de Tabaquismo', ["Unknown", "'never smoked'", "'formerly smoked'", "smokes"])

st.markdown("---")

# 4. Boton de Prediccion y Logica
if st.button('Realizar Prediccion', use_container_width=True):
    with st.spinner("Analizando datos clinicos..."):
        datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
        columnas = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'smoking_status']
        data = pd.DataFrame(datos, columns=columnas)
        
        data_preparada = data.copy()
        
        # Generar variables Dummies
        columnas_categoricas = ['hypertension', 'heart_disease', 'ever_married', 'smoking_status']
        data_preparada = pd.get_dummies(data_preparada, columns=columnas_categoricas, drop_first=False, dtype=int)
        
        # Alinear columnas con el modelo
        data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
        
        # Normalizar las variables numericas
        variables_numericas = ['age', 'avg_glucose_level']
        data_preparada[variables_numericas] = min_max_scaler.transform(data_preparada[variables_numericas])
        
        # Realizar la prediccion
        Y_pred_num = modelo.predict(data_preparada)
        Y_pred_label = labelencoder.inverse_transform(Y_pred_num)
        
        # 5. Mostrar Resultados
        st.header("Resultado de la Evaluacion")
        
        if Y_pred_label[0] == 'Yes':
            st.error("ALERTA: El modelo clasifica al paciente con ALTO RIESGO de ataque al corazon. Se recomienda asistencia e intervencion medica inmediata.")
        else:
            st.success("RESULTADO FAVORABLE: El modelo clasifica al paciente con BAJO RIESGO de ataque al corazon. Mantenga habitos saludables.")
            
        with st.expander("Ver matriz de datos procesada y detalles tecnicos"):
            st.write("Datos transformados enviados al algoritmo:")
            st.dataframe(data_preparada)
            st.info(f"Algoritmo utilizado: {type(modelo).__name__} (Balanceado)")