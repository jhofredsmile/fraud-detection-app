import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Advanced Fraud Detection System")

st.sidebar.header("Enter Transaction Details")

# Inputs
amount = st.sidebar.number_input("Amount", min_value=0.0)
old_balance = st.sidebar.number_input("Old Balance", min_value=0.0)
new_balance = st.sidebar.number_input("New Balance", min_value=0.0)

method = st.sidebar.selectbox(
    "Transaction Method",
    ["UPI", "NFC", "ATM", "NetBanking", "Debit Card", "Credit Card"]
)

lat_from = st.sidebar.number_input("Latitude From")
lon_from = st.sidebar.number_input("Longitude From")

lat_to = st.sidebar.number_input("Latitude To")
lon_to = st.sidebar.number_input("Longitude To")

transactions_24 = st.sidebar.slider("Transactions in last 24 hrs", 1, 30)

hour = st.sidebar.slider("Transaction Hour", 0, 23)
day = st.sidebar.slider("Transaction Day", 1, 31)

is_weekend = st.sidebar.selectbox("Weekend?", [0, 1])
is_new_device = st.sidebar.selectbox("New Device?", [0, 1])
failed_attempts = st.sidebar.slider("Failed Login Attempts", 0, 5)

if st.button("Check Fraud"):

    # Create DataFrame
    input_data = pd.DataFrame([{
        "amount": amount,
        "old_balance": old_balance,
        "new_balance": new_balance,
        "latitude_from": lat_from,
        "longitude_from": lon_from,
        "latitude_to": lat_to,
        "longitude_to": lon_to,
        "transactions_last_24hrs": transactions_24,
        "transaction_hour": hour,
        "transaction_day": day,
        "is_weekend": is_weekend,
        "is_new_device": is_new_device,
        "failed_login_attempts": failed_attempts
    }])

    # Feature Engineering (MUST MATCH TRAINING)
    input_data["amount_ratio"] = input_data["amount"] / (input_data["old_balance"] + 1)
    input_data["balance_diff"] = input_data["old_balance"] - input_data["new_balance"]

    input_data["distance"] = np.sqrt(
        (input_data["latitude_to"] - input_data["latitude_from"])**2 +
        (input_data["longitude_to"] - input_data["longitude_from"])**2
    )

    # One-hot encoding
    input_data = pd.get_dummies(input_data)

    # Add missing columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    # Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Output
    st.subheader("Result")

    st.progress(float(prob))
    st.write(f"Fraud Probability: {prob*100:.2f}%")

    if prediction == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legit Transaction")