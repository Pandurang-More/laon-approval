import streamlit as st
import pickle
import numpy as np

# Load model & preprocessors
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_edu = pickle.load(open("le_edu.pkl", "rb"))
le_emp = pickle.load(open("le_emp.pkl", "rb"))

# Title
st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details:")

# Inputs
dependents = st.number_input("Number of Dependents", 0, 10)

education = st.selectbox(
    "Education",
    ["graduate", "not graduate"]
)

self_emp = st.selectbox(
    "Self Employed",
    ["yes", "no"]
)

income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term (years)")
cibil = st.slider("CIBIL Score", 300, 900)

res_asset = st.number_input("Residential Assets Value")
com_asset = st.number_input("Commercial Assets Value")
lux_asset = st.number_input("Luxury Assets Value")
bank_asset = st.number_input("Bank Asset Value")
# Predict button
if st.button("Predict"):

    try:
        # 🔥 Clean inputs (same as training)
        education_clean = education.strip().lower()
        self_emp_clean = self_emp.strip().lower()

        # Encode
        education_encoded = le_edu.transform([education_clean])[0]
        self_emp_encoded = le_emp.transform([self_emp_clean])[0]

        # Input order MUST match training
        input_data = np.array([[
            dependents,
            education_encoded,
            self_emp_encoded,
            income,
            loan_amount,
            loan_term,
            cibil,
            res_asset,
            com_asset,
            lux_asset, 
            bank_asset    
            
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Output
        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {probability*100:.2f}%)")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {probability*100:.2f}%)")

    except Exception as e:
        st.error(f"Error: {str(e)}")