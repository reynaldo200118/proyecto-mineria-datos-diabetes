import base64, io
import streamlit as st
import pandas as pd
import joblib
from modelo_diabetes_b64 import MODELO_B64

st.set_page_config(page_title="Clasificación académica de diabetes", page_icon="🩺")
st.title("Modelo académico de clasificación de diabetes")
st.warning("Uso exclusivamente académico. No diagnostica ni sustituye la valoración profesional.")
model = joblib.load(io.BytesIO(base64.b64decode(MODELO_B64)))
st.write("Ingresa los valores de las variables para obtener una clasificación estimada.")
with st.form("entrada"):
    a,b=st.columns(2)
    with a:
        pregnancies=st.number_input("Embarazos",0,20,1)
        glucose=st.number_input("Glucosa",0.0,300.0,120.0)
        bp=st.number_input("Presión arterial diastólica",0.0,200.0,70.0)
        skin=st.number_input("Pliegue cutáneo",0.0,100.0,20.0)
    with b:
        insulin=st.number_input("Insulina",0.0,1000.0,80.0)
        bmi=st.number_input("BMI",0.0,80.0,25.0)
        pedigree=st.number_input("Diabetes pedigree function",0.0,3.0,0.5)
        age=st.number_input("Edad",18,120,30)
    go=st.form_submit_button("Evaluar registro")
if go:
    row=pd.DataFrame([{"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,
      "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
      "DiabetesPedigreeFunction":pedigree,"Age":age}])
    pred=int(model.predict(row)[0])
    prob=float(model.predict_proba(row)[0][1])
    st.metric("Probabilidad estimada de clase 1",f"{prob:.1%}")
    if pred: st.error("Clasificación del modelo: clase 1 (positivo).")
    else: st.success("Clasificación del modelo: clase 0 (negativo).")
    st.caption("Estimación estadística, no conclusión médica.")
