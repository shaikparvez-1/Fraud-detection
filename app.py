import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")

st.markdown("""
*Author:* Shaik Parvez  
*Degree:* B.Tech  
*Department:* CSE  
*Institute:* Srinivasa Institute of Technology and Science
""")

@st.cache_data
def load_data():
    return pd.read_csv("fraud_dataset.csv")

data = load_data()

X = data[
    [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
].copy()

y = data["isFraud"]

X["type"] = X["type"].astype("category").cat.codes

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.subheader("Enter Transaction Details")

step = st.number_input("Step", 0)
type_input = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
amount = st.number_input("Amount", 0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", 0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", 0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", 0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", 0.0)

type_encoded = 0 if type_input == "CASH_OUT" else 1

if st.button("Check Fraud"):
    input_data = np.array([[
        step,
        type_encoded,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Safe")