import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Custom CSS (Sky Blue Theme)
st.markdown("""
<style>
body {
    background-color: #e6f2ff;
}
.main {
    background-color: #e6f2ff;
}
h1 {
    text-align: center;
    color: #003366;
    font-size: 50px;
}
.card {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 10px;
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
        method = st.selectbox("Method", ["UPI", "NFC", "ATM", "NetBanking", "Debit Card", "Credit Card"])
        transactions_24 = st.slider("Transactions (24 hrs)", 1, 30)
        hour = st.slider("Hour", 0, 23)

    with col3:
        day = st.slider("Day", 1, 31)
        is_weekend = st.selectbox("Weekend", ["No", "Yes"])
        is_new_device = st.selectbox("New Device", ["No", "Yes"])
        failed_attempts = st.slider("Failed Attempts", 0, 5)

    st.markdown("### 🌍 Location Details")

    col4, col5 = st.columns(2)

    with col4:
        lat_from = st.number_input("From Latitude", value=13.08)
        lon_from = st.number_input("From Longitude", value=80.27)

    with col5:
        lat_to = st.number_input("To Latitude", value=28.61)
        lon_to = st.number_input("To Longitude", value=77.20)

    # Convert
    is_weekend = 1 if is_weekend == "Yes" else 0
    is_new_device = 1 if is_new_device == "Yes" else 0

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

        # Feature Engineering
        input_data["amount_ratio"] = input_data["amount"] / (input_data["old_balance"] + 1)
        input_data["balance_diff"] = input_data["old_balance"] - input_data["new_balance"]
        input_data["distance"] = np.sqrt(
            (input_data["latitude_to"] - input_data["latitude_from"])**2 +
            (input_data["longitude_to"] - input_data["longitude_from"])**2
        )

        input_data = pd.get_dummies(input_data)

        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[columns]

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Fraud Probability")
            st.progress(float(prob))
            st.metric("Risk Score", f"{prob*100:.2f}%")

        with c2:
            st.subheader("⚠️ Risk Level")
            if prob > 0.8:
                st.error("🚨 HIGH RISK")
            elif prob > 0.5:
                st.warning("⚠️ MEDIUM RISK")
            else:
                st.success("✅ LOW RISK")

        st.markdown("### 🌍 Transaction Map")
        map_data = pd.DataFrame({'lat':[lat_from, lat_to], 'lon':[lon_from, lon_to]})
        st.map(map_data)

# ================= TAB 2 =================
with tab2:

    st.markdown("### ⚡ Predefined Fraud / Legit Cases")

    demo_data = [
        ["High Fraud", 90000, 95000, 5000, "Credit Card", 25, 2, 0, 1, 4],
        ["Medium Fraud", 50000, 70000, 20000, "UPI", 15, 3, 0, 1, 3],
        ["Low Fraud", 2000, 50000, 48000, "ATM", 2, 14, 0, 0, 0],
        ["High Fraud", 80000, 85000, 5000, "NFC", 20, 1, 0, 1, 5],
        ["Legit", 1000, 20000, 19000, "Debit Card", 1, 12, 0, 0, 0],
        ["Medium Fraud", 40000, 60000, 20000, "Credit Card", 12, 4, 1, 1, 2],
        ["High Fraud", 95000, 98000, 3000, "UPI", 22, 2, 0, 1, 5],
        ["Legit", 3000, 40000, 37000, "NetBanking", 2, 11, 0, 0, 0],
        ["Medium Fraud", 45000, 70000, 25000, "NFC", 18, 3, 1, 1, 2],
        ["Legit", 1500, 30000, 28500, "ATM", 1, 15, 0, 0, 0]
    ]

    for case in demo_data:
        st.markdown(f"""
        <div class="card">
        <b>{case[0]}</b><br>
        Amount: {case[1]} | Method: {case[4]} | Hour: {case[6]} | Transactions: {case[5]}
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:

    st.markdown("### 🗺️ Transaction Visualizer")

    col1, col2 = st.columns(2)

    with col1:
        lat_from = st.number_input("From Latitude", value=13.08, key="map1")
        lon_from = st.number_input("From Longitude", value=80.27, key="map2")

    with col2:
        lat_to = st.number_input("To Latitude", value=28.61, key="map3")
        lon_to = st.number_input("To Longitude", value=77.20, key="map4")

    distance = np.sqrt((lat_to - lat_from)**2 + (lon_to - lon_from)**2)

    st.metric("Distance", f"{distance:.2f}")

    if distance > 1:
        st.error("🚨 Suspicious Movement")
    elif distance > 0.5:
        st.warning("⚠️ Moderate Movement")
    else:
        st.success("✅ Normal Movement")

    map_data = pd.DataFrame({'lat':[lat_from, lat_to], 'lon':[lon_from, lon_to]})
    st.map(map_data)

# ================= TAB 4 =================
with tab4:

    st.markdown("### 🧠 AI Insights Dashboard")

    try:
        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        st.subheader("📊 Top Features")
        st.bar_chart(imp_df.set_index("Feature"))

    except:
        st.warning("Feature importance not available")

    st.markdown("---")

    st.write("""
🔴 High Risk → Immediate fraud alert  
🟠 Medium Risk → Monitor  
🟢 Low Risk → Safe  

Model considers:
- Transaction amount vs balance
- Location change
- Device risk
- Frequency of transactions
""")