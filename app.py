import streamlit as st
import pickle
import numpy as np

# LOAD TRAINED MODEL
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to check if it is fraudulent.")

# USER INPUTS
step = st.number_input("Step", min_value=0)
type_input = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

# ENCODE TYPE (same logic as training)
type_encoded = 0 if type_input == "CASH_OUT" else 1

# PREDICT
if st.button("Check Fraud"):
    features = np.array(
        [[
            step,
            type_encoded,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
        ]]
    )

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Safe")