import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificación académica de diabetes", page_icon="🩺")
st.title("Modelo académico de clasificación de diabetes")
st.warning("Uso exclusivamente académico. No diagnostica ni sustituye la valoración profesional.")

model = json.loads(Path(__file__).with_name("modelo_arbol.json").read_text(encoding="utf-8"))
features = model["features"]
zero_cols = set(model["zero_cols"])
medians = model["medians"]
tree = model["tree"]

st.write("Introduce los valores para obtener una clasificación estimada.")
with st.form("datos"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Embarazos", 0, 20, 1)
        glucose = st.number_input("Glucosa", 0.0, 300.0, 120.0)
        bp = st.number_input("Presión arterial diastólica", 0.0, 200.0, 70.0)
        skin = st.number_input("Pliegue cutáneo", 0.0, 100.0, 20.0)
    with c2:
        insulin = st.number_input("Insulina", 0.0, 1000.0, 80.0)
        bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
        pedigree = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5)
        age = st.number_input("Edad", 18, 120, 30)
    submit = st.form_submit_button("Evaluar registro")

if submit:
    row = {"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,
           "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
           "DiabetesPedigreeFunction":pedigree,"Age":age}
    x = []
    for i, name in enumerate(features):
        value = float(row[name])
        if name in zero_cols and value == 0:
            value = float(medians[i])
        x.append(value)
    node = 0
    while tree["left"][node] != -1:
        feature_index = tree["feature"][node]
        if x[feature_index] <= tree["threshold"][node]:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    probability = float(tree["value"][node])
    prediction = int(probability >= 0.5)
    st.metric("Probabilidad estimada de clase 1", f"{probability:.1%}")
    if prediction == 1:
        st.error("Clasificación estimada: clase 1 (positivo).")
    else:
        st.success("Clasificación estimada: clase 0 (negativo).")
    st.caption("Estimación estadística académica; no constituye diagnóstico médico.")
