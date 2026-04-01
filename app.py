import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. BLOOMBERG TERMINAL UI STYLING ---
st.set_page_config(page_title="TERMINAL: YCS", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #cccccc; }
    h1, h2, h3 { color: #ff9900 !important; font-family: 'Courier New', monospace; text-transform: uppercase; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #333333; }
    [data-testid="stMetricLabel"] { color: #00ff00 !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Courier New', monospace; }
    .stTable { border: 1px solid #333333; background-color: #000000; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    # ^IRX (13-week), ^TNX (10-year), ^TYX (30-year)
    tickers = {"^IRX": "3M_BILL", "^TNX": "10Y_YIELD", "^TYX": "30Y_BOND"}
    raw = yf.download(list(tickers.keys()), period="10y", interval="1d")
    
    # Handle MultiIndex and drop NaNs for Scikit-Learn stability
    df = raw['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_data()
    
    # --- 3. PREDICTIVE AI MODEL (OLS REGRESSION) ---
    # Target: 10Y Yield | Features: Short-term policy (3M) and Long-term inflation (30Y)
    X = df[['3M_BILL', '30Y_BOND']]
    y = df['10Y_YIELD']
    model = LinearRegression().fit(X, y)

    # --- 4. TERMINAL HEADER ---
    st.title("BBG: YIELD_CURVE_PREDICT <GO>")
    st.markdown("---")

    # --- 5. SIDEBAR: MACRO "WHAT-IF" ANALYSIS ---
    st.sidebar.markdown("### <CMD> PARAMETERS")
    last_3m = df['3M_BILL'].iloc[-1]
    last_30y = df['30Y_BOND'].iloc[-1]

    st.sidebar.info(f"MKT 3M: {last_3m:.2f}% | MKT 30Y: {last_30y:.2f}%")
    
    # Shocks in Basis Points (bps)
    short_shock = st.sidebar.slider("SHOCK: FED POLICY (BPS)", -200, 200, 0) / 100
    long_shock = st.sidebar.slider("SHOCK: INF/GROWTH (BPS)", -200, 200, 0) / 100

    # Predictive Calculation for User Scenario
    sim_x = np.array([[last_3m + short_shock, last_30y + long_shock]])
    fair_value = model.predict(sim_x)[0]

    # --- 6. MAIN DASHBOARD METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("10Y MARKET", f"{df['10Y_YIELD'].iloc[-1]:.3f}%")
    c2.metric("10Y FAIR VAL", f"{fair_value:.3f}%", delta=f"{fair_value - df['10Y_YIELD'].iloc[-1]:.3f}%")
    c3.metric("MODEL R-SQR", f"{model.score(X, y):.4f}")
    
    # Residual identifies if current bond is Rich or Cheap
    current_resid = df['10Y_YIELD'].iloc[-1] - model.predict([[last_3m, last_30y]])[0]
    c4.metric("RICH/CHEAP", f"{current_resid*100:.2f} bps")

    # --- 7. SENSITIVITY MATRIX (<YCS> FUNCTION) ---
    st.markdown("### <YCS> SENSITIVITY_MATRIX")
    
    s_range = np.linspace(last_3m - 0.5, last_3m + 0.5, 7)
    l_range = np.linspace(last_30y - 0.5, last_30y + 0.5, 7)
    
    matrix = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
    matrix_df = pd.DataFrame(matrix, index=[f"{s:.2f}" for s in s_range], columns=[f"{l:.2f}" for l in l_range])
    
    st.table(matrix_df.style.format("{:.3f}")
             .background_gradient(cmap='magma', axis=None)
             .set_properties(**{'background-color': 'black', 'color': 'white'}))

    st.markdown("`SYSTEM STATUS: MODEL_NOMINAL | DATA_CURRENT`")

except Exception as e:
    st.error(f"TERMINAL ERROR: {e}")
