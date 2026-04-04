import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Credit Card Fraud Detection")

# User Inputs
amount = st.number_input("Transaction Amount")
old_balance = st.number_input("Old Balance")
new_balance = st.number_input("New Balance")

method = st.selectbox("Transaction Method", ["UPI", "Card Tap", "ATM", "NetBanking"])

lat = st.number_input("Latitude")
lon = st.number_input("Longitude")

if st.button("Check Fraud"):
    # Create DataFrame
    input_data = pd.DataFrame([[amount, old_balance, new_balance, lat, lon]],
                              columns=['amount','old_balance','new_balance','latitude','longitude'])

    # Add missing columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Encode method
    input_data[f"transaction_method_{method}"] = 1

    input_data = input_data[columns]

    # Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Legit Transaction. Probability: {prob:.2f}")