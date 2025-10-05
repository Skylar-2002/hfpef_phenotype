import streamlit as st
import pandas as pd
import joblib

st.header("HFpEF Phenotyping Tool")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    creatinine = st.number_input("Creatinine", min_value=0.0, value=1.0, step=0.1)
    bun = st.number_input("BUN (Blood Urea Nitrogen)", min_value=0.0, value=15.0, step=0.1)
    chloride = st.number_input("Chloride", min_value=0.0, value=100.0, step=0.1)
    mch = st.number_input("MCH (Mean Corpuscular Hemoglobin)", min_value=0.0, value=28.0, step=0.1)
    aniongap = st.number_input("Anion Gap", min_value=0.0, value=12.0, step=0.1)
    dbp = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0, value=80, step=1)
    mchc = st.number_input("MCHC (Mean Corpuscular Hemoglobin Concentration)", min_value=0.0, value=34.0, step=0.1)

with col2:
    mbp = st.number_input("Mean Blood Pressure (MBP)", min_value=0, value=90, step=1)
    resp_rate = st.number_input("Respiratory Rate", min_value=0, value=16, step=1)
    sbp = st.number_input("Systolic Blood Pressure (SBP)", min_value=0, value=120, step=1)
    platelet = st.number_input("Platelet Count", min_value=0, value=250, step=1)
    calcium = st.number_input("Calcium", min_value=0.0, value=9.0, step=0.1)
    diabetes = st.selectbox("Diabetes", ("No", "Yes"))
    renal_disease = st.selectbox("Renal Disease", ("No", "Yes"))

# 编码分类变量
diabetes_encoded = 1 if diabetes == "Yes" else 0
renal_disease_encoded = 1 if renal_disease == "Yes" else 0

try:
    model = joblib.load('clf.pkl')
    
    X = pd.DataFrame([[
        creatinine, bun, chloride, mch, aniongap, dbp, mchc, mbp,
        diabetes_encoded, resp_rate, sbp, renal_disease_encoded,
        platelet, calcium
    ]], columns=[
        "creatinine", "bun", "chloride", "mch", "aniongap", "dbp", "mchc", "mbp",
        "diabetes", "resp_rate", "sbp", "renal_disease", 
        "platelet", "calcium"
    ])
    
    if st.button("Predict"):
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        st.success(f"Prediction: {prediction[0]}")
        st.info(f"Probability: {probability[0][1]:.2f}")
    
    with st.expander("View Input Summary"):
        st.write(X)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure the model file 'clf.pkl' exists and all inputs are valid.")