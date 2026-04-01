import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. TERMINAL UI STYLING ---
st.set_page_config(page_title="TERMINAL: YCS", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #cccccc; }
    h1, h2, h3 { color: #ff9900 !important; font-family: 'Courier New', monospace; text-transform: uppercase; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #333333; }
    [data-testid="stMetricLabel"] { color: #00ff00 !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Courier New', monospace; }
    .stChart { border: 1px solid #333333; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    tickers = {"^IRX": "3M_BILL", "^TNX": "10Y_YIELD", "^TYX": "30Y_BOND"}
    raw = yf.download(list(tickers.keys()), period="5y", interval="1d")
    df = raw['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_data()
    
    # AI Engine
    X = df[['3M_BILL', '30Y_BOND']]
    y = df['10Y_YIELD']
    model = LinearRegression().fit(X, y)

    # --- 3. HEADER & METRICS ---
    st.title("BBG: YIELD_CURVE_PREDICT <GO>")
    
    c1, c2, c3, c4 = st.columns(4)
    last_10y = df['10Y_YIELD'].iloc[-1]
    fair_val = model.predict([[df['3M_BILL'].iloc[-1], df['30Y_BOND'].iloc[-1]]])[0]
    
    c1.metric("10Y MKT", f"{last_10y:.3f}%")
    c2.metric("10Y FAIR", f"{fair_val:.3f}%", delta=f"{fair_val - last_10y:.3f}%")
    c3.metric("MODEL R2", f"{model.score(X, y):.4f}")
    c4.metric("SPREAD", f"{(df['30Y_BOND'].iloc[-1] - df['3M_BILL'].iloc[-1])*100:.1f} bps")

    st.markdown("---")

    # --- 4. INTERACTIVE LINE CHART (<GP> FUNCTION) ---
    st.subheader("<GP> HISTORICAL YIELD COMPARISON")
    
    # We use st.line_chart for a clean, interactive Bloomberg-style look
    # Pre-selecting colors that fit the terminal aesthetic
    chart_data = df[['3M_BILL', '10Y_YIELD', '30Y_BOND']]
    st.line_chart(chart_data, height=400, use_container_width=True)

    

    # --- 5. SIDEBAR: WHAT-IF SCENARIOS ---
    st.sidebar.markdown("### <CMD> PARAMETERS")
    short_shock = st.sidebar.slider("SHOCK: 3M BILL (BPS)", -200, 200, 0) / 100
    long_shock = st.sidebar.slider("SHOCK: 30Y BOND (BPS)", -200, 200, 0) / 100

    # Predicted outcome based on shocks
    sim_input = np.array([[df['3M_BILL'].iloc[-1] + short_shock, df['30Y_BOND'].iloc[-1] + long_shock]])
    scenario_val = model.predict(sim_input)[0]
    
    st.sidebar.markdown("---")
    st.sidebar.metric("SCENARIO 10Y PREDICTION", f"{scenario_val:.3f}%")

    # --- 6. SENSITIVITY MATRIX ---
    st.markdown("### <YCS> SENSITIVITY_MATRIX")
    s_range = np.linspace(df['3M_BILL'].iloc[-1] - 0.5, df['3M_BILL'].iloc[-1] + 0.5, 7)
    l_range = np.linspace(df['30Y_BOND'].iloc[-1] - 0.5, df['30Y_BOND'].iloc[-1] + 0.5, 7)
    
    grid = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
    matrix_df = pd.DataFrame(grid, index=[f"{s:.2f}" for s in s_range], columns=[f"{l:.2f}" for l in l_range])
    
    st.table(matrix_df.style.format("{:.3f}").background_gradient(cmap='magma', axis=None))

except Exception as e:
    st.error(f"TERMINAL ERROR: {e}")
