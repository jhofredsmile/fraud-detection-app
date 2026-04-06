import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# CSS Theme
st.markdown("""
<style>
body {background-color: #e6f2ff;}
h1 {text-align:center; color:#003366; font-size:50px;}
.card {
    background-color:white;
    padding:15px;
    border-radius:15px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom:10px;
}
.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("fraud_model.pkl")
columns = joblib.load("columns.pkl")

# Title
st.markdown("<h1>🏦 AI Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Enter Transaction",
    "⚡ Demo Cases",
    "🗺️ Transaction Visualizer",
    "🧠 AI Insights"
])

# ================= TAB 1 =================
with tab1:
    st.markdown("### 💼 Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Amount", 0.0)
        old_balance = st.number_input("Old Balance", 0.0)
        new_balance = st.number_input("New Balance", 0.0)

    with col2:
        method = st.selectbox("Method", ["UPI","NFC","ATM","NetBanking","Debit Card","Credit Card"])
        transactions_24 = st.slider("Transactions (24 hrs)", 1, 30)
        hour = st.slider("Hour", 0, 23)

    with col3:
        day = st.slider("Day", 1, 31)
        is_weekend = st.selectbox("Weekend", ["No","Yes"])
        is_new_device = st.selectbox("New Device", ["No","Yes"])
        failed_attempts = st.slider("Failed Attempts", 0, 5)

    st.markdown("### 🌍 Location Details")

    col4, col5 = st.columns(2)
    with col4:
        lat_from = st.number_input("From Latitude", value=13.08)
        lon_from = st.number_input("From Longitude", value=80.27)
    with col5:
        lat_to = st.number_input("To Latitude", value=28.61)
        lon_to = st.number_input("To Longitude", value=77.20)

    is_weekend = 1 if is_weekend=="Yes" else 0
    is_new_device = 1 if is_new_device=="Yes" else 0

    if st.button("🚀 Analyze Transaction"):

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

        input_data["amount_ratio"] = input_data["amount"]/(input_data["old_balance"]+1)
        input_data["balance_diff"] = input_data["old_balance"] - input_data["new_balance"]
        input_data["distance"] = np.sqrt(
            (input_data["latitude_to"]-input_data["latitude_from"])**2 +
            (input_data["longitude_to"]-input_data["longitude_from"])**2
        )

        input_data = pd.get_dummies(input_data)

        for col in columns:
            if col not in input_data.columns:
                input_data[col]=0

        input_data = input_data[columns]

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.markdown(f"<div class='result-box' style='background:#ff4d4d;'>🚨 FRAUD DETECTED<br>{prob*100:.2f}% Risk</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box' style='background:#66cc66;'>✅ LEGIT TRANSACTION<br>{prob*100:.2f}% Risk</div>", unsafe_allow_html=True)

        st.progress(float(prob))

        st.map(pd.DataFrame({'lat':[lat_from,lat_to],'lon':[lon_from,lon_to]}))

# ================= TAB 2 =================
with tab2:

    st.markdown("### ⚡ Click Demo Case")

    demo_cases = [
        ("High Fraud",90000,95000,5000,"Credit Card",25,2,0,1,4,13.08,80.27,28.61,77.20),
        ("Medium Fraud",50000,70000,20000,"UPI",15,3,0,1,3,19.07,72.87,22.57,88.36),
        ("Legit",2000,50000,48000,"ATM",2,14,0,0,0,13.08,80.27,13.10,80.30)
    ]

    for i,case in enumerate(demo_cases):

        if st.button(f"{case[0]} Case {i+1}"):

            with st.spinner("Analyzing..."):
                time.sleep(1)

            input_data = pd.DataFrame([{
                "amount": case[1],
                "old_balance": case[2],
                "new_balance": case[3],
                "latitude_from": case[10],
                "longitude_from": case[11],
                "latitude_to": case[12],
                "longitude_to": case[13],
                "transactions_last_24hrs": case[5],
                "transaction_hour": case[6],
                "transaction_day": 15,
                "is_weekend": case[7],
                "is_new_device": case[8],
                "failed_login_attempts": case[9]
            }])

            input_data["amount_ratio"] = input_data["amount"]/(input_data["old_balance"]+1)
            input_data["balance_diff"] = input_data["old_balance"] - input_data["new_balance"]
            input_data["distance"] = np.sqrt(
                (input_data["latitude_to"]-input_data["latitude_from"])**2 +
                (input_data["longitude_to"]-input_data["longitude_from"])**2
            )

            input_data = pd.get_dummies(input_data)

            for col in columns:
                if col not in input_data.columns:
                    input_data[col]=0

            input_data = input_data[columns]

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.markdown(f"<div class='result-box' style='background:#ff4d4d;'>🚨 FRAUD<br>{prob*100:.2f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-box' style='background:#66cc66;'>✅ LEGIT<br>{prob*100:.2f}%</div>", unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:

    st.markdown("### 🗺️ Distance (KM)")

    lat1 = st.number_input("Lat From", value=13.08)
    lon1 = st.number_input("Lon From", value=80.27)
    lat2 = st.number_input("Lat To", value=28.61)
    lon2 = st.number_input("Lon To", value=77.20)

    # Haversine formula
    R = 6371
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)

    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

    distance_km = R*c

    st.metric("Distance (KM)", f"{distance_km:.2f} km")

    if distance_km > 1500:
        st.error("🚨 Impossible Travel Detected")
    elif distance_km > 500:
        st.warning("⚠️ Suspicious Travel")
    else:
        st.success("✅ Normal Travel")

    st.map(pd.DataFrame({'lat':[lat1,lat2],'lon':[lon1,lon2]}))

# ================= TAB 4 =================
with tab4:

    st.markdown("### 🧠 AI Insights")

    try:
        imp = pd.DataFrame({
            "Feature": columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        st.bar_chart(imp.set_index("Feature"))
    except:
        st.write("No feature importance available")