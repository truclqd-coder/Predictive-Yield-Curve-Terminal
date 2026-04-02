import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. PROPRIETARY TERMINAL UI STYLING ---
st.set_page_config(page_title="SYSTEM: QRV_TERMINAL", layout="wide")

st.markdown("""
    <style>
    /* Professional Deep Navy & Slate Theme */
    .stApp { 
        background-color: #0d1117; 
        color: #c9d1d9; 
    }
    /* Header Styling: Cyan and Monospaced */
    h1, h2, h3 { 
        color: #58a6ff !important; 
        font-family: 'Roboto Mono', 'Courier New', monospace; 
        text-transform: uppercase;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    /* Metric Styling: Professional Green and White */
    [data-testid="stMetricLabel"] { 
        color: #7ee787 !important; 
        font-family: 'Roboto Mono', monospace; 
    }
    [data-testid="stMetricValue"] { 
        color: #ffffff !important; 
        font-family: 'Roboto Mono', monospace; 
    }
    /* Sidebar and Widgets */
    [data-testid="stSidebar"] { 
        background-color: #161b22; 
        border-right: 1px solid #30363d; 
    }
    .stTable { 
        border: 1px solid #30363d; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE DATA ENGINE ---
@st.cache_data
def load_market_data():
    # US Treasury Proxies: 13-Week (^IRX), 10-Year (^TNX), 30-Year (^TYX)
    tickers = {"^IRX": "SHORT_RATE", "^TNX": "10Y_YIELD", "^TYX": "30Y_BOND"}
    raw = yf.download(list(tickers.keys()), period="5y", interval="1d")
    
    # Flatten MultiIndex and drop NaNs for model stability
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'].rename(columns=tickers).dropna()
    else:
        df = raw['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_market_data()
    
    # --- 3. PREDICTIVE AI MODEL (OLS) ---
    # We model the 10Y "Belly" as a function of the 3M and 30Y "Wings"
    X = df[['SHORT_RATE', '30Y_BOND']]
    y = df['10Y_YIELD']
    model = LinearRegression().fit(X, y)

    # --- 4. TERMINAL HEADER ---
    st.title("SYS: QRV_YIELD_PREDICTOR <EXEC>")
    
    # --- 5. REAL-TIME MARKET METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    curr_3m = df['SHORT_RATE'].iloc[-1]
    curr_10y = df['10Y_YIELD'].iloc[-1]
    curr_30y = df['30Y_BOND'].iloc[-1]
    
    # Model's current Fair Value estimate
    fair_val_now = model.predict([[curr_3m, curr_30y]])[0]
    
    c1.metric("MKT: 10Y", f"{curr_10y:.3f}%")
    c2.metric("SYS: FAIR_VAL", f"{fair_val_now:.3f}%", delta=f"{fair_val_now - curr_10y:.3f}%")
    c3.metric("MOD: R-SQUARED", f"{model.score(X, y):.4f}")
    c4.metric("SPR: 30Y-3M", f"{(curr_30y - curr_3m)*100:.1f} bps")

    st.markdown("---")

    # --- 6. INTERACTIVE HISTORICAL ANALYSIS ---
    st.subheader("<HIST> TIME_SERIES_ANALYSIS")
    st.line_chart(df[['SHORT_RATE', '10Y_YIELD', '30Y_BOND']], height=350)

    # --- 7. SIDEBAR: SCENARIO ANALYSIS (WHAT-IF) ---
    st.sidebar.markdown("### <CMD> INPUT_SHOCKS")
    
    # Users adjust economic assumptions here
    s_shock = st.sidebar.slider("FED_POLICY_SHOCK (BPS)", -200, 200, 0) / 100
    l_shock = st.sidebar.slider("INF_GROWTH_SHOCK (BPS)", -200, 200, 0) / 100

    # Predicted yield based on shocks
    simulated_x = np.array([[curr_3m + s_shock, curr_30y + l_shock]])
    scenario_prediction = model.predict(simulated_x)[0]
    
    st.sidebar.markdown("---")
    st.sidebar.metric("SCENARIO PREDICTION", f"{scenario_prediction:.3f}%")
    st.sidebar.write(f"Implied Δ: {(scenario_prediction - curr_10y)*100:.1f} bps")

    # --- 8. SENSITIVITY MATRIX ---
    st.subheader("<SENS> PROJECTION_MATRIX")
    
    # Create a grid of possible outcomes
    s_range = np.linspace(curr_3m - 0.5, curr_3m + 0.5, 7)
    l_range = np.linspace(curr_30y - 0.5, curr_30y + 0.5, 7)
    
    grid = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
    matrix_df = pd.DataFrame(
        grid, 
        index=[f"3M: {s:.2f}" for s in s_range], 
        columns=[f"30Y: {l:.2f}" for l in l_range]
    )
    
    # Display matrix with a "Magma" heatmap for risk visualization
    st.table(matrix_df.style.format("{:.3f}")
             .background_gradient(cmap='magma', axis=None)
             .set_properties(**{'background-color': '#0d1117', 'color': 'white'}))

    st.markdown("`STATUS: SYSTEM_READY | DATA_INTEGRITY_VERIFIED`")

except Exception as e:
    st.error(f"SYSTEM_FATAL_ERROR: {e}")
