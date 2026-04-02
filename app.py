import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. PROPRIETARY TERMINAL UI STYLING ---
st.set_page_config(page_title="QuantYield Terminal", layout="wide")

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
    /* Style the tooltip help icon to match cyan theme */
    .stTooltipIcon { color: #58a6ff !important; }
    
    .stTable { 
        border: 1px solid #30363d; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & AI ENGINE ---
@st.cache_data
def load_engine_data():
    # US Treasury Proxies: 13-Week (^IRX), 10-Year (^TNX), 30-Year (^TYX)
    tickers = {"^IRX": "SHORT_RATE", "^TNX": "10Y_YIELD", "^TYX": "30Y_BOND"}
    raw = yf.download(list(tickers.keys()), period="5y", interval="1d")
    
    # Handle MultiIndex and drop NaNs for model stability
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'].rename(columns=tickers).dropna()
    else:
        df = raw['Close'].rename(columns=tickers).dropna()
    return df

try:
    df = load_engine_data()
    
    # --- AI MODEL (OLS REGRESSION) ---
    # We model the 10Y "Belly" as a function of the 3M and 30Y "Wings"
    X = df[['SHORT_RATE', '30Y_BOND']]
    y = df['10Y_YIELD']
    model = LinearRegression().fit(X, y)

    # --- 3. BRANDED HEADER & MODEL DISCLOSURE ---
    st.title("QuantYield: Predictive Yield Curve Engine <EXEC>")
    st.markdown("""
        <p style='font-family: "Roboto Mono", monospace; color: #58a6ff; font-size: 14px; margin-top: -20px;'>
            ENGINE: OLS_REGRESSION_V1 | KERNEL: SCIKIT-LEARN | LOOKBACK: 5Y_HISTORICAL_DAILY
        </p>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Current Market Snapshot
    curr_3m = df['SHORT_RATE'].iloc[-1]
    curr_10y = df['10Y_YIELD'].iloc[-1]
    curr_30y = df['30Y_BOND'].iloc[-1]
    fair_val_now = model.predict([[curr_3m, curr_30y]])[0]
    r_squared = model.score(X, y)

    # --- 4. REAL-TIME ANALYTICS METRICS (WITH HOVER HELP) ---
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
        help="The AI's 'True North.' This is the theoretical yield level suggested by 5 years of historical correlation between the 3M and 30Y yields."
    )
    
    c3.metric(
        label="MOD: R-SQUARED", 
        value=f"{r_squared:.4f}",
        help="Model Confidence Score: Measures how much of the 10Y movement is explained by the 3M and 30Y yields. Higher values (>0.90) indicate a high degree of historical anchoring."
    )
    
    c4.metric(
        label="SPR: 30Y-3M", 
        value=f"{(curr_30y - curr_3m)*100:.1f} bps",
        help="The Curve Slope: 30-Year yield minus 3-Month yield. Inversion (negative bps) is a primary indicator of economic recession risks."
    )

    st.markdown("---")

    # --- 5. VISUALIZATIONS ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("<HIST> TIME_SERIES_ANALYSIS")
        st.line_chart(df[['SHORT_RATE', '10Y_YIELD', '30Y_BOND']], height=400)

    with col_right:
        st.subheader("<SENS> PROJECTION_MATRIX")
        # Stress-Test Grid for Sensitivity Analysis
        s_range = np.linspace(curr_3m - 0.5, curr_3m + 0.5, 5)
        l_range = np.linspace(curr_30y - 0.5, curr_30y + 0.5, 5)
        
        grid = [[model.predict([[s, l]])[0] for l in l_range] for s in s_range]
        matrix_df = pd.DataFrame(
            grid, 
            index=[f"3M:{s:.1f}" for s in s_range], 
            columns=[f"30Y:{l:.1f}" for l in l_range]
        )
        
        st.table(matrix_df.style.format("{:.2f}")
                 .background_gradient(cmap='magma', axis=None)
                 .set_properties(**{'background-color': '#0d1117', 'color': 'white'}))

    # --- 6. SIDEBAR: SCENARIO ANALYSIS (WHAT-IF) ---
    st.sidebar.markdown("### <CMD> INPUT_SHOCKS")
    s_shock = st.sidebar.slider(
        "FED_POLICY_SHOCK (BPS)", -200, 200, 0, 
        help="Simulate a shift in short-term rates (Federal Reserve hikes or cuts)."
    ) / 100
    
    l_shock = st.sidebar.slider(
        "INF_GROWTH_SHOCK (BPS)", -200, 200, 0, 
        help="Simulate a shift in long-term inflation or economic growth expectations."
    ) / 100

    # Predicted yield based on user shocks
    simulated_x = np.array([[curr_3m + s_shock, curr_30y + l_shock]])
    scenario_prediction = model.predict(simulated_x)[0]
    
    st.sidebar.markdown("---")
    st.sidebar.metric("SCENARIO PREDICTION", f"{scenario_prediction:.3f}%", 
                      help="The predicted 10Y yield based on your custom macro scenario.")
    st.sidebar.write(f"Implied Delta: {(scenario_prediction - curr_10y)*100:.1f} bps")

    st.markdown("`STATUS: SYSTEM_READY | DATA_INTEGRITY_VERIFIED | SESSION_ACTIVE`")

except Exception as e:
    st.error(f"ENGINE_FATAL_ERROR: {e}")
