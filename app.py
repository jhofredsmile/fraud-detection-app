import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import sqlite3

# ---------------- DATABASE ----------------
conn = sqlite3.connect('app.db', check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    amount REAL,
    method TEXT,
    result TEXT,
    probability REAL
)''')
conn.commit()

def add_user(u,p):
    c.execute("INSERT INTO users VALUES (?,?)",(u,p))
    conn.commit()

def login_user(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p))
    return c.fetchone()

def add_history(user, amt, method, res, prob):
    c.execute("INSERT INTO history VALUES (?,?,?,?,?)",(user,amt,method,res,prob))
    conn.commit()

def get_history(user):
    c.execute("SELECT * FROM history WHERE username=?",(user,))
    return c.fetchall()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.markdown("<h1 style='text-align:center;'>🔐 Secure Login</h1>", unsafe_allow_html=True)

    tabL, tabS = st.tabs(["Login","Sign Up"])

    with tabL:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u,p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.success("Welcome "+u)
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tabS:
        nu = st.text_input("New Username")
        npw = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            add_user(nu,npw)
            st.success("Account created!")

    st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("fraud_model.pkl")
columns = joblib.load("columns.pkl")

st.markdown("<h1>🏦 Fraud Detection System</h1>", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Enter Transaction","Demo Cases","Distance","AI Insights","History"
])

# ---------------- COMMON FUNCTION ----------------
def predict(data):
    df = pd.DataFrame([data])
    df["amount_ratio"] = df["amount"]/(df["old_balance"]+1)
    df["balance_diff"] = df["old_balance"] - df["new_balance"]
    df["distance"] = np.sqrt(
        (df["latitude_to"]-df["latitude_from"])**2 +
        (df["longitude_to"]-df["longitude_from"])**2
    )

    df = pd.get_dummies(df)
    for col in columns:
        if col not in df.columns:
            df[col]=0
    df = df[columns]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return pred, prob

# ---------------- TAB 1 ----------------
with tab1:

    amount = st.number_input("Amount")
    old = st.number_input("Old Balance")
    new = st.number_input("New Balance")
    method = st.selectbox("Method",["UPI","NFC","ATM","NetBanking"])
    t24 = st.slider("Transactions 24hr",1,30)
    hour = st.slider("Hour",0,23)

    lat1 = st.number_input("From Lat", value=13.08)
    lon1 = st.number_input("From Lon", value=80.27)
    lat2 = st.number_input("To Lat", value=28.61)
    lon2 = st.number_input("To Lon", value=77.20)

    if st.button("Analyze"):

        data = {
            "amount":amount,"old_balance":old,"new_balance":new,
            "latitude_from":lat1,"longitude_from":lon1,
            "latitude_to":lat2,"longitude_to":lon2,
            "transactions_last_24hrs":t24,
            "transaction_hour":hour,"transaction_day":10,
            "is_weekend":0,"is_new_device":1,"failed_login_attempts":2
        }

        pred, prob = predict(data)

        res = "FRAUD" if pred==1 else "LEGIT"

        st.success(f"{res} ({prob*100:.2f}%)")

        add_history(st.session_state.user, amount, method, res, prob)

# ---------------- TAB 2 ----------------
with tab2:

    cases = [
        ("Fraud",90000,95000,5000,"UPI",22,2,13.08,80.27,28.61,77.20),
        ("Legit",2000,50000,48000,"ATM",2,14,13.08,80.27,13.10,80.30)
    ]

    for i,c in enumerate(cases):

        if st.button(f"Case {i+1} ({c[0]})"):

            with st.spinner("Processing..."):
                time.sleep(1)

            data = {
                "amount":c[1],"old_balance":c[2],"new_balance":c[3],
                "latitude_from":c[7],"longitude_from":c[8],
                "latitude_to":c[9],"longitude_to":c[10],
                "transactions_last_24hrs":c[5],
                "transaction_hour":c[6],"transaction_day":10,
                "is_weekend":0,"is_new_device":1,"failed_login_attempts":3
            }

            pred, prob = predict(data)

            res = "FRAUD" if pred==1 else "LEGIT"

            st.error(res) if pred==1 else st.success(res)

            st.write("### Details")
            st.write(data)

            add_history(st.session_state.user, c[1], c[4], res, prob)

# ---------------- TAB 3 ----------------
with tab3:

    lat1 = st.number_input("Lat1", value=13.08)
    lon1 = st.number_input("Lon1", value=80.27)
    lat2 = st.number_input("Lat2", value=28.61)
    lon2 = st.number_input("Lon2", value=77.20)

    R=6371
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)

    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

    dist = R*c

    st.metric("Distance KM", f"{dist:.2f}")

# ---------------- TAB 4 ----------------
with tab4:
    st.write("AI insights based on features")

# ---------------- TAB 5 ----------------
with tab5:

    st.markdown("### 📜 Transaction History")

    hist = get_history(st.session_state.user)

    if hist:
        df = pd.DataFrame(hist, columns=["User","Amount","Method","Result","Probability"])
        st.dataframe(df)
    else:
        st.write("No history yet")