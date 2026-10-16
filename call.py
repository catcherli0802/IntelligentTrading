import streamlit as st
import requests
import numpy as np
from scipy.stats import norm
from datetime import datetime
import re

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮")
st.title("🧮 Option Delta Calculator")
st.write("Calculates analytical Black-Scholes Call Delta using the official Alpaca Data API feed.")

# --- SECURE CREDENTIAL INPUT PANEL ---
# Stripped hardcoded keys to prevent account security vulnerabilities
ALPACA_KEY_ID = st.sidebar.text_input("Alpaca API Key ID", type="password")
ALPACA_SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", type="password")

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
            # 1. Fetch current stock spot price using the single asset endpoint layout
            stock_url = f"https://alpaca.markets{ticker_sym}/snapshot"
            stock_res = requests.get(stock_url, headers=headers).json()
            
            if not stock_res or ("latestTrade" not in stock_res and "minuteBar" not in stock_res):
                st.error(f"Could not retrieve stock market price for '{ticker_sym}'. Ensure the ticker symbol is valid.")
                st.stop()
            
            # Fallback chain to handle missing trades during off-market windows
            current_price = None
            if "latestTrade" in stock_res and stock_res["latestTrade"]:
                current_price = stock_res["latestTrade"].get("p")
            if not current_price and "minuteBar" in stock_res and stock_res["minuteBar"]:
                current_price = stock_res["minuteBar"].get("c")
                
            if not current_price:
                st.error(f"No current pricing data available for '{ticker_sym}' right now.")
                st.stop()
                
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
            options_url = f"https://alpaca.markets{ticker_sym}"
            options_res = requests.get(options_url, headers=headers).json()
            option_snapshots = options_res.get("snapshots", {})
            
            sigma = None
            closest_diff = float('inf')
            actual_strike = strike_input
            
            # Format comparison string target (e.g. "261120" for Nov 20, 2026)
            target_date_string = target_expiry.strftime("%y%m%d")
            
            # 4. Filter the OPRA option stream array securely via precise regex matching
            # Matches text pattern: [Ticker][6 Digits Date][C/P][8 Digits Strike]
            pattern = re.compile(rf"^{ticker_sym}{target_date_string}C(\d+)$")
            
            for symbol, data in option_snapshots.items():
                match = pattern.match(symbol)
                if match:
                    try:
                        # Safely extract explicitly parsed strike value via regex matching group
                        strike_raw = match.group(1)
                        strike_val = float(strike_raw) / 1000.0
                        
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
                sigma = 0.25  # Global baseline asset volatility floor proxy
                st.warning(f"Could not locate a valid contract with IV for {maturity_input}. Fallback to standard 25.0% asset IV.")
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
            st.error(f"Data stream mapping exception: {str(global_err)}. Alpaca servers may be updating.")
