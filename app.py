import streamlit as st
import pandas as pd
import joblib

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="HFpEF Phenotype Mapping Tool",
    layout="centered"
)

# =========================================================
# Title & description (Reviewer-friendly)
# =========================================================
st.title("HFpEF Phenotype Mapping Tool")
st.caption(
    "This tool assigns HFpEF phenotypes using a supervised TabPFN classifier trained to "
    "reproduce unsupervised K-prototypes-derived phenotypes. "
    "It is intended for phenotype mapping rather than outcome prediction."
)

st.divider()

# =========================================================
# Input layout
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Laboratory Measurements")
    creatinine = st.number_input(
        "Creatinine (mg/dL)", min_value=0.0, value=1.0, step=0.1
    )
    bun = st.number_input(
        "Blood Urea Nitrogen (BUN, mg/dL)", min_value=0.0, value=15.0, step=1.0
    )
    chloride = st.number_input(
        "Chloride (mmol/L)", min_value=0.0, value=100.0, step=1.0
    )
    hematocrit = st.number_input(
        "Hematocrit (%)", min_value=0.0, value=40.0, step=1.0
    )
    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=0.0, value=120.0, step=5.0
    )
    calcium = st.number_input(
        "Calcium (mg/dL)", min_value=0.0, value=9.0, step=0.1
    )
    aniongap = st.number_input(
        "Anion Gap (mmol/L)", min_value=0.0, value=12.0, step=1.0
    )
    platelet = st.number_input(
        "Platelet Count (×10³/µL)", min_value=0.0, value=250.0, step=10.0
    )

with col2:
    st.subheader("Vital Signs")
    mbp = st.number_input(
        "Mean Blood Pressure (MBP, mmHg)", min_value=0.0, value=90.0, step=1.0
    )
    sbp = st.number_input(
        "Systolic Blood Pressure (SBP, mmHg)", min_value=0.0, value=120.0, step=1.0
    )
    heart_rate = st.number_input(
        "Heart Rate (beats/min)", min_value=0, value=80, step=1
    )
    resp_rate = st.number_input(
        "Respiratory Rate (breaths/min)", min_value=0, value=16, step=1
    )
    spo2 = st.number_input(
        "Oxygen Saturation (SpO₂, %)", min_value=0.0, max_value=100.0, value=98.0, step=1.0
    )
    
st.subheader("Comorbidity")
renal_disease = st.selectbox("History of Renal Disease", ("No", "Yes"))
renal_disease_encoded = 1 if renal_disease == "Yes" else 0

st.divider()

# =========================================================
# Load trained TabPFN model
# =========================================================
try:
    model = joblib.load("clf.pkl")
except Exception as e:
    st.error(f"Failed to load model file (clf.pkl): {e}")
    st.stop()

# =========================================================
# Prepare input dataframe (STRICT feature order)
# =========================================================
X = pd.DataFrame([[
    mbp,
    creatinine,
    bun,
    resp_rate,
    chloride,
    sbp,
    heart_rate,
    renal_disease_encoded,
    hematocrit,
    glucose,
    calcium,
    spo2,
    aniongap,
    platelet
]], columns=[
    "mbp",
    "creatinine",
    "bun",
    "resp_rate",
    "chloride",
    "sbp",
    "heart_rate",
    "renal_disease",
    "hematocrit",
    "glucose",
    "calcium",
    "spo2",
    "aniongap",
    "platelet"
])

# =========================================================
# Phenotype descriptions (official, paper-consistent)
# =========================================================
phenotype_description = {
    1: (
        "Phenotype 1 – Diabetic and Renal Phenotype: "
        "Characterized by severe renal impairment and metabolic acidosis, "
        "this phenotype exhibited the highest 1-year mortality. "
    ),
    2: (
        "Phenotype 2 – Hypertensive and Pulmonary Phenotype: "
        "Marked by cardiovascular and pulmonary comorbidities with "
        "hemodynamic instability. "
    ),
    3: (
        "Phenotype 3 – Low-Risk, Low Blood Pressure and Arrhythmia Phenotype: "
        "Despite advanced age, patients demonstrated preserved physiological stability "
        "and the most favorable prognosis. Diuretics consistently showed significant "
        "survival benefit across all cohorts."
    )
}

# =========================================================
# Phenotype mapping
# =========================================================
if st.button("Assign HFpEF Phenotype"):
    prediction = model.predict(X)
    probability = model.predict_proba(X)

    phenotype_id = int(prediction[0])  # expected: 1 / 2 / 3
    confidence = probability[0][phenotype_id - 1]  # adjust for 0-based index

    st.markdown("### Phenotype Assignment Result")
    st.success(f"Assigned HFpEF Phenotype: Phenotype {phenotype_id}")

    st.info(f"Phenotype assignment confidence: {confidence:.2f}")

    st.markdown("### Phenotype Interpretation")
    st.write(
        phenotype_description.get(
            phenotype_id,
            "No phenotype description available."
        )
    )

    st.divider()

    st.caption(
        "⚠️ This tool assigns HFpEF phenotypes using a supervised classifier trained "
        "to reproduce unsupervised K-prototypes-derived phenotypes. "
        "It does not predict clinical outcomes, prognosis, or treatment response."
    )

# =========================================================
# Optional: display input summary
# =========================================================
with st.expander("View Input Data Summary"):
    st.dataframe(X)
