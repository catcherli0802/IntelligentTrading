import streamlit as st
import requests
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮")
st.title("🧮 Option Delta Calculator")
st.write("Calculates analytical Black-Scholes Call Delta using the official Alpaca Data API feed.")

# --- DYNAMIC HARDCODED API FEED CREDENTIALS ---
ALPACA_KEY_ID = st.sidebar.text_input("Alpaca API Key ID", value="PKT6PY6DEOHNZZQKWPFERDU7FS", type="password")
ALPACA_SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", value="3s6Sz9Usc9yV6eGo5Wt7wnGYYVFgyzZ3Dq5dqtrFVyF7", type="password")

# --- CORE MATH ENGINE ---
def bs_call_delta(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.5000
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- USER PARAMETERS ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker_sym = st.text_input("Stock Ticker", value="AAPL").upper().strip()
with col2:
    strike_input = st.number_input("Target Strike Price ($)", value=340.0, step=5.0)
with col3:
    # Added field back so you can input your specific maturity timeline
    maturity_input = st.text_input("Maturity Date (YYYY-MM-DD)", value="2026-11-20")

if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    st.info("💡 Please enter your free Alpaca API Key credentials in the sidebar to activate the live feed.")
else:
    headers = {
        "Apca-Api-Key-Id": ALPACA_KEY_ID,
        "Apca-Api-Secret-Key": ALPACA_SECRET_KEY
    }
    
    if st.button("Calculate Delta"):
        try:
            # 1. Fetch current delayed stock spot price from Alpaca
            stock_url = f"https://data.alpaca.markets/v2/stocks/{ticker_sym}/snapshots"
            stock_res = requests.get(stock_url, headers=headers).json()
            
            if "snapshot" not in stock_res or not stock_res["snapshot"]:
                st.error(f"Could not retrieve stock market price for '{ticker_sym}'. Ensure the ticker symbol is valid.")
            else:
                current_price = stock_res["snapshot"]["latestTrade"]["p"]
                
                # 2. Parse the target user maturity date string
                try:
                    today = datetime.now().date()
                    target_expiry = datetime.strptime(maturity_input.strip(), "%Y-%m-%d").date()
                    days_to_maturity = (target_expiry - today).days
                    
                    if days_to_maturity <= 0:
                        st.error("Maturity date must be a date in the future.")
                        st.stop()
                except ValueError:
                    st.error("Incorrect date format string. Please use YYYY-MM-DD (e.g. 2026-11-20).")
                    st.stop()
                
                # 3. Fetch options chain array snapshots
                options_url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{ticker_sym}"
                options_res = requests.get(options_url, headers=headers).json()
                option_snapshots = options_res.get("snapshots", {})
                
                sigma = None
                closest_diff = float('inf')
                actual_strike = strike_input
                
                # Format our comparison match target string (e.g. "261120" for Nov 20, 2026)
                target_date_string = target_expiry.strftime("%y%m%d")
                
                # 4. Filter the OPRA option stream array for our target date and strike
                for symbol, data in option_snapshots.items():
                    # Symbol format example: AAPL261120C00340000
                    if target_date_string in symbol and "C" in symbol:
                        try:
                            # Extract strike price embedded directly inside the contract string name
                            strike_part = symbol.split("C")[-1]
                            strike_val = float(strike_part) / 1000.0
                            
                            diff = abs(strike_val - strike_input)
                            if diff < closest_diff:
                                greeks = data.get("greeks", {})
                                iv = greeks.get("implied_volatility")
                                
                                if iv and iv > 0:
                                    closest_diff = diff
                                    actual_strike = strike_val
                                    sigma = iv
                        except:
                            continue
                
                # 5. Process Engine Fallbacks or Output Variables
                if sigma is None:
                    sigma = 0.25 # Global baseline asset volatility floor proxy
                    st.warning(f"Could not locate an exact option contract for {maturity_input}. Fallback to standard 25.0% historical asset IV.")
                elif closest_diff > 0.01:
                    st.info(f"Exact strike contract unavailable. Core engine snapped to nearest available market strike: ${actual_strike:.2f}")

                # Institutional risk-free bond yield floor proxy (4.50%)
                r = 0.045
                T = days_to_maturity / 365.0
                
                # Run the quantitative Black-Scholes calculation
                delta = bs_call_delta(current_price, actual_strike, r, sigma, T)
                
                # --- RENDER WEB UI PANEL ---
                st.markdown("---")
                st.subheader("📊 Financial Dashboard Results")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Stock Price", f"${current_price:.2f}")
                res_col2.metric("Market IV (σ)", f"{sigma*100:.1f}%")
                res_col3.metric("Days to Expiry", f"{days_to_maturity} Days")
                res_col4.metric("Call Delta (Δ)", f"{delta:.4f}")
                
        except Exception as global_err:
            st.error("Data stream mapping exception. Alpaca servers may be updating. Please retry in a few seconds.")
