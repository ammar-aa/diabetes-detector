import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(page_title="Diabetes Prediction System", page_icon="Consultant", layout="wide")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('diabetes/diabetes_model.keras')
    with open('diabetes/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("Diabetes Prediction System")
st.write("Please fill in the health details below:")

age_map = {
    1.0: "18-24 years old",
    2.0: "25-29 years old",
    3.0: "30-34 years old",
    4.0: "35-39 years old",
    5.0: "40-44 years old",
    6.0: "45-49 years old",
    7.0: "50-54 years old",
    8.0: "55-59 years old",
    9.0: "60-64 years old",
    10.0: "65-69 years old",
    11.0: "70-74 years old",
    12.0: "75-79 years old",
    13.0: "80 years or older"
}

col1, col2, col3 = st.columns(3)

with col1:
    high_bp = st.selectbox("High Blood Pressure", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    high_chol = st.selectbox("High Cholesterol", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    chol_check = st.selectbox("Cholesterol Check (Last 5 Years)", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    smoker = st.selectbox("Smoker (100+ Cigarettes in Lifetime)", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    stroke = st.selectbox("Stroke History", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    heart_disease = st.selectbox("Heart Disease or Attack", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    phys_activity = st.selectbox("Physical Activity (Last 30 Days)", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    fruits = st.selectbox("Eat Fruits Daily", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    veggies = st.selectbox("Eat Vegetables Daily", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    hvy_alcohol = st.selectbox("Heavy Alcohol Consumption", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    any_healthcare = st.selectbox("Has Healthcare Coverage", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    no_doc_cost = st.selectbox("Financial Barrier to See Doctor", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    gen_hlth = st.slider("General Health Rating (1-Excellent, 5-Poor)", 1.0, 5.0, 3.0)

with col3:
    ment_hlth = st.number_input("Mental Health (Days unwell/month)", 0.0, 30.0, 0.0)
    phys_hlth = st.number_input("Physical Health (Days unwell/month)", 0.0, 30.0, 0.0)
    diff_walk = st.selectbox("Difficulty Walking", [0.0, 1.0], format_func=lambda x: "Yes" if x==1 else "No")
    sex = st.selectbox("Sex", [0.0, 1.0], format_func=lambda x: "Male" if x==1 else "Female")
    age_display = st.selectbox("Age Category", options=list(age_map.values()))
    age = [k for k, v in age_map.items() if v == age_display][0]
    education = st.slider("Education Level (1-6)", 1.0, 6.0, 4.0)
    income = st.slider("Income Level (1-8)", 1.0, 8.0, 5.0)

if st.button("Analyze Result", use_container_width=True):
    features = np.array([[
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_activity, fruits, veggies, hvy_alcohol, any_healthcare, no_doc_cost,
        gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
    ]])
    
    features_scaled = scaler.transform(features)
    prediction_prob = model.predict(features_scaled)
    prediction = (prediction_prob > 0.45).astype("int32")
    
    st.divider()
    if prediction[0][0] == 1:
        st.error(f"Prediction: Potential Risk of Diabetes. (Confidence: {prediction_prob[0][0]:.2%})")
    else:
        st.success(f"Prediction: No Significant Risk Detected. (Confidence: {1 - prediction_prob[0][0]:.2%})")
