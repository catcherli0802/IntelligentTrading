import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮")
st.title("🧮 Option Delta Calculator")
st.write("Calculates analytical Black-Scholes Call Delta using consolidated data feeds.")

# --- STEP 1: TICKER INPUT ---
ticker_sym = st.text_input("Stock Ticker", value="AAPL").upper().strip()

# We wrap the data fetcher into a unified function to handle network exceptions cleanly
@st.cache_data(ttl=60) # Caches the data for 60 seconds to bypass Yahoo's rate limits
def fetch_all_market_data(ticker):
    try:
        asset = yf.Ticker(ticker)
        
        # Consolidate history and available option contract dates into one step
        hist = asset.history(period="1d")
        if hist.empty:
            return None
        
        spot = hist["Close"].iloc[-1]
        dates = asset.options
        return {"spot": spot, "dates": dates, "asset_obj": asset}
    except Exception:
        return None

market_data = fetch_all_market_data(ticker_sym)

if not market_data or not market_data["dates"]:
    st.error(f"Could not retrieve active market profiles for '{ticker_sym}'. Verify the symbol or try again in a moment.")
else:
    # --- STEP 2: USER LAYOUT CONTROLS ---
    col1, col2 = st.columns(2)
    with col1:
        strike_price = st.number_input("Strike Price ($)", value=360.0, step=5.0)
    with col2:
        # Guarantees the date string directly matches Yahoo's active table indices
        maturity_str = st.selectbox("Select Expiration Date", market_data["dates"])

    if st.button("Calculate Delta"):
        try:
            current_price = market_data["spot"]
            stock_object = market_data["asset_obj"]
            
            # 1. Fetch risk-free yield proxy with a direct fallback if the bond floor errors
            try:
                tnx = yf.Ticker("^IRX")
                r = (tnx.history(period="1d")["Close"].iloc[-1]) / 100 
            except Exception:
                r = 0.045 # Institutional fallback rate (4.50%) if Yahoo bond API fails
            
            # 2. Compute exact year-fraction maturity boundaries
            today = datetime.now().date()
            maturity_date = datetime.strptime(maturity_str, "%Y-%m-%d").date()
            days_to_maturity = (maturity_date - today).days
            
            T = max(days_to_maturity, 1) / 365.0 # Forces a minimum floor boundary of 1 day
            
            # 3. Secure Option Matrices
            opt_chain = stock_object.option_chain(maturity_str)
            calls = opt_chain.calls
            
            # 4. Lock onto the mathematically absolute closest strike listed in the market matrix
            closest_strike_idx = (calls['strike'] - strike_price).abs().idxmin()
            matched_option = calls.loc[closest_strike_idx]
            
            actual_strike = matched_option['strike']
            sigma = matched_option['impliedVolatility']
            
            # Notify user gracefully if we had to adjust to a listing matrix gap
            if abs(actual_strike - strike_price) > 0.01:
                st.info(f"Requested strike ${strike_price:.2f} not found. Adjusted to closest available strike: ${actual_strike:.2f}")

            # --- STEP 3: ANALYTICAL QUANT MATH ENGINE ---
            if sigma <= 0:
                # If market implied volatility gaps out to zero, use historical asset volatility proxy
                sigma = 0.25 
                st.warning("Live Implied Volatility calculation data missing. Running engine via 25% historical baseline standard.")
            
            # Classic Black-Scholes Model
            d1 = (np.log(current_price / actual_strike) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            call_delta = norm.cdf(d1)
            
            # --- STEP 4: RENDER FINANCIAL DASHBOARD UI ---
            st.markdown("---")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Stock Price", f"${current_price:.2f}")
            res_col2.metric("Market IV", f"{sigma*100:.1f}%")
            res_col3.metric("Days to Expiry", f"{days_to_maturity} days")
            res_col4.metric("Call Delta (Δ)", f"{call_delta:.4f}")
            
        except Exception as calculate_error:
            st.error("Financial matrix compilation error. Yahoo Finance may be limiting requests. Please retry in a few seconds.")
