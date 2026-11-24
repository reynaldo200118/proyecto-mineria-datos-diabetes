import base64
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

class ZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = self.columns if self.columns is not None else X.columns
        for col in cols:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X

st.set_page_config(page_title="Clasificación académica de diabetes", page_icon="🩺")
st.title("Modelo académico de clasificación de diabetes")
st.warning("Uso exclusivamente académico. No diagnostica ni sustituye la valoración profesional.")
try:
    encoded = Path(__file__).with_name("modelo_diabetes.b64").read_text(encoding="ascii")
    model = joblib.load(io.BytesIO(base64.b64decode(encoded)))
except Exception as exc:
    st.error("No se pudo cargar el modelo. Revisa los archivos y dependencias.")
    st.exception(exc)
    st.stop()

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
    row = pd.DataFrame([{"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,
        "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
        "DiabetesPedigreeFunction":pedigree,"Age":age}])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    st.metric("Probabilidad estimada de clase 1", f"{prob:.1%}")
    if pred == 1:
        st.error("Clasificación estimada: clase 1 (positivo).")
    else:
        st.success("Clasificación estimada: clase 0 (negativo).")
    st.caption("Estimación estadística académica; no constituye diagnóstico médico.")
