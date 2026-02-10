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

# LOAD DATASET
data = pd.read_csv("fraud_dataset.csv")

# PREPARE DATA
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

# ENCODE TYPE
X.loc[:, "type"] = X["type"].astype("category").cat.codes

# TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.write("Enter transaction details:")

# USER INPUTS
step = st.number_input("Step", min_value=0)
type_input = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

type_encoded = 0 if type_input == "CASH_OUT" else 1

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