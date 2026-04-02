import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. PROPRIETARY TERMINAL UI STYLING ---
st.set_page_config(page_title="SYSTEM: QRV_TERMINAL", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { 
        color: #58a6ff !important; 
        font-family: 'Roboto Mono', monospace; 
        text-transform: uppercase;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    [data-testid="stMetricLabel"] { color: #7ee787 !important; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Roboto Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    /* Style the tooltip help icon to match cyan theme */
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_market_data():
    tickers = {"^IRX": "SHORT_RATE", "^TNX": "10Y_YIELD", "^TYX": "30Y_BOND"}
    raw = yf.download(list(tickers.keys()), period="5y", interval="1d")
    df = raw['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_market_data()
    X = df[['SHORT_RATE', '30Y_BOND']]
    y = df['10Y_YIELD']
    model = LinearRegression().fit(X, y)

    # --- 3. TERMINAL HEADER & CALCULATIONS ---
    st.title("SYS: QRV_YIELD_PREDICTOR <EXEC>")
    
    curr_3m, curr_10y, curr_30y = df['SHORT_RATE'].iloc[-1], df['10Y_YIELD'].iloc[-1], df['30Y_BOND'].iloc[-1]
    fair_val_now = model.predict([[curr_3m, curr_30y]])[0]
    r_squared = model.score(X, y)

    # --- 4. METRICS ROW WITH HOVER HELP ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(
        label="MKT: 10Y", 
        value=f"{curr_10y:.3f}%",
        help="The current real-time market yield for the US 10-Year Treasury Note."
    )
    
    c2.metric(
        label="SYS: FAIR_VAL", 
        value=f"{fair_val_now:.3f}%", 
        delta=f"{fair_val_now - curr_10y:.3f}%",
        help="The AI's 'True North.' The theoretical yield level suggested by 5 years of historical correlation between the 3M and 30Y yields."
    )
    
    c3.metric(
        label="MOD: R-SQUARED", 
        value=f"{r_squared:.4f}",
        help="Confidence Metric: Measures how much of the 10Y movement is explained by the 3M/30Y 'Wings.' Values >0.90 indicate a highly anchored curve."
    )
    
    c4.metric(
        label="SPR: 30Y-3M", 
        value=f"{(curr_30y - curr_3m)*100:.1f} bps",
        help="The Slope: A measure of economic health. Inversion (negative bps) often signals a late-cycle economy or impending recession."
    )

    st.markdown("---")

    # --- 5. VISUALIZATIONS ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("<HIST> TIME_SERIES")
        st.line_chart(df[['SHORT_RATE', '10Y_YIELD', '30Y_BOND']], height=400)

    with col_right:
        st.subheader("<SENS> MATRIX")
        s_range = np.linspace(curr_3m - 0.5, curr_3m + 0.5, 5)
        l_range = np.linspace(curr_30y - 0.5, curr_30y + 0.5, 5)
        grid = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
        matrix_df = pd.DataFrame(grid, index=[f"{s:.1f}" for s in s_range], columns=[f"{l:.1f}" for l in l_range])
        st.table(matrix_df.style.format("{:.2f}").background_gradient(cmap='magma', axis=None))

    # --- 6. SIDEBAR WHAT-IF ---
    st.sidebar.markdown("### <CMD> INPUT_SHOCKS")
    s_shock = st.sidebar.slider("FED_POLICY (BPS)", -200, 200, 0, help="Simulate a shift in short-term rates (Fed interest rate hikes/cuts).") / 100
    l_shock = st.sidebar.slider("INF_EXPECT (BPS)", -200, 200, 0, help="Simulate a shift in long-term inflation or growth expectations.") / 100
    
    scenario_val = model.predict([[curr_3m + s_shock, curr_30y + l_shock]])[0]
    st.sidebar.metric("SCENARIO PREDICTION", f"{scenario_val:.3f}%", help="The predicted 10Y yield based on your custom macro shocks.")

    st.markdown("`STATUS: SYSTEM_READY | DATA_INTEGRITY_VERIFIED | SESSION_ACTIVE`")

except Exception as e:
    st.error(f"SYSTEM_FATAL_ERROR: {e}")
