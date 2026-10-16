import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮")
st.title("🧮 Option Delta Calculator")
st.write("Calculates analytical Black-Scholes Call Delta using direct manual input fields.")

# --- CORE MATH ENGINE ---
def bs_call_delta(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.5000
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- USER PARAMETERS PANEL ---
st.subheader("📝 Model Input Parameters")

col1, col2 = st.columns(2)
with col1:
    current_price = st.number_input("Underlying Stock Price ($)", value=240.00, min_value=0.01, step=1.0, help="The current trading spot price of the stock.")
    strike_input = st.number_input("Target Strike Price ($)", value=245.00, min_value=0.01, step=1.0, help="The option contract execution strike price.")

with col2:
    iv_percent = st.number_input("Market Implied Volatility (σ %)", value=25.0, min_value=0.1, max_value=500.0, step=0.5, help="The expected future volatility percentage priced into the option.")
    maturity_input = st.text_input("Maturity Date (YYYY-MM-DD)", value="2026-11-20", help="The expiration timeline target for the option contract.")

# Institutional risk-free bond yield baseline configuration proxy (4.50%)
r = 0.045

if st.button("Calculate Analytical Delta"):
    try:
        # 1. Parse the target user maturity timeline date string
        today = datetime.now().date()
        target_expiry = datetime.strptime(maturity_input.strip(), "%Y-%m-%d").date()
        days_to_maturity = (target_expiry - today).days
        
        if days_to_maturity <= 0:
            st.error("Maturity date must be a date in the future.")
            st.stop()
            
        # Convert inputs to exact core engineering float variables
        sigma = iv_percent / 100.0
        T = days_to_maturity / 365.0
        
        # 2. Run the quantitative analytical Black-Scholes engine calculation
        delta = bs_call_delta(current_price, strike_input, r, sigma, T)
        
        # --- RENDER DASHBOARD VISUAL PANEL ---
        st.markdown("---")
        st.subheader("📊 Financial Dashboard Results")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Stock Price", f"${current_price:.2f}")
        res_col2.metric("Market IV (σ)", f"{iv_percent:.1f}%")
        res_col3.metric("Days to Expiry", f"{days_to_maturity} Days")
        res_col4.metric("Call Delta (Δ)", f"{delta:.4f}")
        
    except ValueError:
        st.error("Incorrect date format string. Please use YYYY-MM-DD exactly (e.g., 2026-11-20).")
    except Exception as err:
        st.error(f"Mathematical execution exception: {str(err)}")
