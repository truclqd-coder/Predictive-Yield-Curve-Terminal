import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. TERMINAL UI STYLING ---
st.set_page_config(page_title="QuantYield Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1 { color: #58a6ff !important; font-family: 'Roboto Mono', monospace; margin-bottom: 0px; }
    h3 { color: #58a6ff !important; font-family: 'Roboto Mono', monospace; border-bottom: 1px solid #30363d; }
    [data-testid="stMetricLabel"] { color: #7ee787 !important; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Roboto Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stTable { border: 1px solid #30363d; background-color: #0d1117; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    # 3-Month (^IRX), 10-Year (^TNX), 30-Year (^TYX)
    tickers = {"^IRX": "3M", "^TNX": "10Y", "^TYX": "30Y"}
    df = yf.download(list(tickers.keys()), period="5y", interval="1d")['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_data()
    X = df[['3M', '30Y']]
    y = df['10Y']
    # The Core AI Instrument: OLS_REGRESSION_V1
    model = LinearRegression().fit(X, y)

    # --- 3. BRANDED HEADER & INSTRUMENT STATUS ---
    st.title("QuantYield: Predictive Yield Curve Engine <EXEC>")
    
    st.markdown("""
        <p style='font-family: "Roboto Mono", monospace; color: #58a6ff; font-size: 14px; margin-top: 5px; line-height: 1.6;'>
            <span style='color: #7ee787;'>●</span> <b>AI Engine Status:</b> OLS_REGRESSION_V1 is active. 
            Utilizing Ordinary Least Squares to synthesize historical priors with user-defined calibration.
        </p>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # --- 4. CORE ANALYTICS METRICS (WITH ENHANCED HELP) ---
    c1, c2, c3, c4 = st.columns(4)
    curr_3m, curr_10y, curr_30y = df['3M'].iloc[-1], df['10Y'].iloc[-1], df['30Y'].iloc[-1]
    fair_val = model.predict([[curr_3m, curr_30y]])[0]

    c1.metric(
        label="MKT: 10Y", 
        value=f"{curr_10y:.3f}%", 
        help="Current real-time market yield for the 10-Year Treasury Note. This is the 'Live' benchmark."
    )
    
    c2.metric(
        label="SYS: FAIR_VAL", 
        value=f"{fair_val:.3f}%", 
        delta=f"{fair_val - curr_10y:.3f}%", 
        help="The AI's 'True North.' Derived by OLS_REGRESSION_V1 using 5 years of historical correlation between short-end policy and long-end growth."
    )
    
    c3.metric(
        label="MOD: R-SQUARED", 
        value=f"{model.score(X, y):.4f}", 
        help="Confidence metric: Indicates how much 10Y variance is explained by the 3M and 30Y yields. Higher values suggest a tighter structural anchor."
    )
    
    c4.metric(
        label="SPR: 30Y-3M", 
        value=f"{(curr_30y - curr_3m)*100:.1f} bps", 
        help="The Slope: A fundamental macro indicator. Measures the yield spread between the long-bond and the short-rate in Basis Points (bps)."
    )

    st.markdown("---")

    # --- 5. VISUAL ANALYSIS ---
    col_chart, col_sens = st.columns([2, 1])

    with col_chart:
        st.subheader("<HIST> TIME_SERIES_ANALYSIS")
        st.line_chart(df, height=400)

    with col_sens:
        st.subheader("<SENS> PROJECTION_MATRIX")
        s_range = np.linspace(curr_3m - 0.5, curr_3m + 0.5, 5)
        l_range = np.linspace(curr_30y - 0.5, curr_30y + 0.5, 5)
        grid = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
        matrix_df = pd.DataFrame(grid, index=[f"3M:{s:.1f}%" for s in s_range], columns=[f"30Y:{l:.1f}%" for l in l_range])
        st.table(matrix_df.style.format("{:.2f}").background_gradient(cmap='magma', axis=None))
        st.caption("Matrix units: Predicted 10Y Yield (%)")

    # --- 6. SIDEBAR: INSTRUMENT CALIBRATION ---
    st.sidebar.markdown("### <SYS> AI_CALIBRATION")
    st.sidebar.markdown("""
        <p style='font-family: monospace; font-size: 12px; color: #7ee787;'>
            INSTRUMENT: OLS_REGRESSION_V1<br>
            MODE: MANUAL_SHOCK_SYNTHESIS
        </p>
        """, unsafe_allow_html=True)
    
    st.sidebar.info("""
        **CALIBRATION GUIDE:**
        1. **Short-Rate:** Adjust for expected Federal Reserve policy shifts.
        2. **Long-Bond:** Adjust for changes in inflation or growth outlook.
        3. **Synthesis:** Recalculate Fair Value based on these new priors.
    """)
    
    st.sidebar.divider()
    
    s_shock = st.sidebar.slider("SHORT_RATE_SHOCK (BPS)", -200, 200, 0) / 100
    l_shock = st.sidebar.slider("LONG_BOND_SHOCK (BPS)", -200, 200, 0) / 100
    
    new_pred = model.predict([[curr_3m + s_shock, curr_30y + l_shock]])[0]
    st.sidebar.divider()
    
    # ENHANCED POSTERIOR HELP
    st.sidebar.metric(
        label="POSTERIOR PREDICTION", 
        value=f"{new_pred:.3f}%",
        help="The synthesized yield expectation. This combines 5 years of historical evidence (Priors) with your custom macro adjustments (Evidence)."
    )
    
    st.sidebar.write(f"Implied Delta: {(new_pred - curr_10y)*100:.1f} bps")
    st.sidebar.info("Prediction generated via OLS_REGRESSION_V1 synthesis.")

    st.markdown("`STATUS: SYSTEM_READY | OLS_REGRESSION_V1_ENABLED | SESSION_ACTIVE`")

except Exception as e:
    st.error(f"System Error: {e}")
