import streamlit as st
import joblib
import numpy as np

# 1. Saved model ko load karna
model = joblib.load('heart_disease_model.pkl')

# 2. App ki Heading
st.title("❤️ Heart Disease Prediction App")
st.write("Yeh app Decision Tree model par bani hai. Apni medical details enter karein:")

# 3. User se inputs lene ke liye fields (Heart dataset ke saare 12 features)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", value=250)
diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", value=38)
high_blood_pressure = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
platelets = st.number_input("Platelets", value=263000.0)
serum_creatinine = st.number_input("Serum Creatinine", value=1.1)
serum_sodium = st.number_input("Serum Sodium", value=137)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
time = st.number_input("Follow-up period (days)", value=130)

# 4. Predict Button
if st.button("Predict Heart Risk"):
    # Input data ko saare 12 features ke sath array mein convert karna
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                            high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])
    
    # Model se prediction lena
    prediction = model.predict(input_data)
    
    # Result show karna
    if prediction[0] == 1:
        st.error("⚠️ Result: High risk of heart failure event detected.")
    else:
        st.success("✅ Result: Low risk / Patient is stable.")