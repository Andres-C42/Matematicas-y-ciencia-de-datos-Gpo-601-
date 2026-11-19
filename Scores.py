import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la calificación  ''')
st.image("Cali.png", caption="Predicción de la calificación.")

st.header('Datos del estudiante')

def user_input_features():
  horas_estudio = st.number_input('Horas de Estudio:', min_value=1, max_value=12, value = 1, step = 1)
  horas_sueno = st.number_input('Horas de Sueño:',  min_value=4, max_value=10, value = 4, step = 1)
  asistencia = st.number_input('Porcentaje de Asistencia:', min_value=0, max_value=100, value = 0, step = 1)
  calificacion_previa = st.number_input('Calificacion previa:', min_value=0, max_value=100, value = 0, step = 1)

  user_input_data = {'hours_studied': horas_estudio,
                     'sleep_hours': horas_sueno,
                     'attendance_percent': asistencia,
                     'previous_scores': calificacion_previa,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('student_exam_scores_df.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Predicción de la calificación')
st.write('La calificación del estudiante es ', prediccion)
