import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮", layout="wide")
st.title("🧮 Option Greeks Calculator")
st.write("Calculates analytical Black-Scholes Call option Greeks using direct manual input fields.")

# --- CORE QUANTITATIVE MATH ENGINE ---
def calculate_greeks(S, K, r, sigma, T):
    """
    Computes analytical Black-Scholes formulas for a European Call Option.
    Returns: (Delta, Gamma, Vega, Theta, Rho)
    """
    if T <= 0 or sigma <= 0:
        return 0.5000, 0.0000, 0.0000, 0.0000, 0.0000
        
    # Intermediate mathematical components
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    # 1. Delta (Δ)
    delta = cdf_d1
    
    # 2. Gamma (Γ)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    
    # 3. Vega (ν)
    # Scaled by /100 to show the option dollar change per 1% absolute move in Implied Volatility
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0
    
    # 4. Theta (θ)
    # Scaled by /365 to show the option dollar decay per 1 calendar day step forward in time
    theta_term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    theta_term2 = -r * K * np.exp(-r * T) * cdf_d2
    theta = (theta_term1 + theta_term2) / 365.0
    
    # 5. Rho (ρ)
    # Scaled by /100 to show the option dollar change per 1% absolute move in interest rates
    rho = (K * T * np.exp(-r * T) * cdf_d2) / 100.0
    
    return delta, gamma, vega, theta, rho

# --- ROW 1: USER INPUT PARAMETERS ---
st.subheader("📝 Model Input Parameters")
in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns(5)

with in_col1:
    ticker_sym = st.text_input("Stock Ticker", value="AAPL", help="Asset symbol for tracking context.").upper().strip()
with in_col2:
    current_price = st.number_input("Stock Price ($)", value=240.00, min_value=0.01, step=1.0, help="Current trading spot price of the stock (S).")
with in_col3:
    strike_input = st.number_input("Target Strike ($)", value=245.00, min_value=0.01, step=1.0, help="Option contract execution strike price (K).")
with in_col4:
    iv_percent = st.number_input("Implied Volatility (σ %)", value=25.0, min_value=0.1, max_value=500.0, step=0.5, help="Expected future volatility percentage (sigma).")
with in_col5:
    maturity_input = st.text_input("Maturity (YYYY-MM-DD)", value="2026-11-20", help="Expiration timeline target for the option contract.")

# Institutional risk-free bond yield baseline configuration proxy (4.50% / r = 0.045)
r = 0.045

st.markdown("---")

# Execute calculations automatically based on inputs or via a dedicated form-level layout
try:
    # 1. Parse the target user maturity timeline date string
    today = datetime.now().date()
    target_expiry = datetime.strptime(maturity_input.strip(), "%Y-%m-%d").date()
    days_to_maturity = (target_expiry - today).days
    
    if days_to_maturity <= 0:
        st.error("Maturity date must be a date in the future.")
    else:
        # Convert inputs to exact core engineering float variables
        sigma = iv_percent / 100.0
        T = days_to_maturity / 365.0
        
        # 2. Run the math calculations
        delta, gamma, vega, theta, rho = calculate_greeks(current_price, strike_input, r, sigma, T)
        
        # Convert Delta to an easily readable percentage form
        delta_percentage = delta * 100
        
        # --- ROW 2: DASHBOARD VISUAL PANEL ---
        st.subheader(f"📊 Financial Dashboard Results ({ticker_sym}) — {days_to_maturity} Days to Expiry")
        
        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
        
        with res_col1:
            st.metric("Delta (Δ)", f"{delta_percentage:.2f}%", help="Sensitivity to asset price shifts. Expressed here as equivalent hedge ratio percentage.")
        with res_col2:
            st.metric("Gamma (Γ)", f"{gamma:.4f}", help="Acceleration rate of Delta. Change in Delta per $1 stock move.")
        with res_col3:
            st.metric("Vega (ν)", f"{vega:.4f}", help="Sensitivity to volatility. Dollar option change per 1% move in Volatility.")
        with res_col4:
            st.metric("Theta (θ)", f"{theta:.4f}", help="Time decay metric. Dollar option value erosion per 1 passing day.")
        with res_col5:
            st.metric("Rho (ρ)", f"{rho:.4f}", help="Sensitivity to interest rates. Dollar option change per 1% move in macro rates.")
            
except ValueError:
    st.error("Incorrect date format string. Please use YYYY-MM-DD exactly (e.g., 2026-11-20).")
except Exception as err:
    st.error(f"Mathematical execution exception: {str(err)}")
