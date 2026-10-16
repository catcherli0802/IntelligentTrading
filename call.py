import streamlit as st
import requests
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Intelligent Alpha Trading", page_icon="🧮")
st.title("🧮 Option Delta Calculator")
st.write("Calculates analytical Black-Scholes Call Delta using the official Alpaca Data API feed.")

# --- SET YOUR KEYS HERE ---
ALPACA_KEY_ID = st.sidebar.text_input("Alpaca API Key ID", value="PKT6PY6DEOHNZZQKWPFERDU7FS", type="password")
ALPACA_SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", value="3s6Sz9Usc9yV6eGo5Wt7wnGYYVFgyzZ3Dq5dqtrFVyF7", type="password")

def bs_call_delta(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.5000
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Step 1: User Inputs
ticker_sym = st.text_input("Stock Ticker", value="AAPL").upper().strip()
strike_input = st.number_input("Target Strike Price ($)", value=340.0, step=5.0)

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
            stock_url = f"https://alpaca.markets{ticker_sym}/snapshots"
            stock_res = requests.get(stock_url, headers=headers).json()
            
            if "snapshot" not in stock_res or not stock_res["snapshot"]:
                st.error("Invalid ticker symbol or data feed error.")
            else:
                current_price = stock_res["snapshot"]["latestTrade"]["p"]
                
                # 2. Fetch the options chain layout matrix
                # Note: Alpaca options tracking formats symbols as 'AAPL261120C00340000'
                options_url = f"https://alpaca.markets{ticker_sym}"
                options_res = requests.get(options_url, headers=headers).json()
                
                # Filter down to extract expiration date vectors and IV matrices
                option_snapshots = options_res.get("snapshots", {})
                
                valid_contracts = []
                for symbol, data in option_snapshots.items():
                    # Parse out standard details from contract keys (parsing rules for OPRA)
                    # We look for call options close to user configurations
                    if "C" in symbol:  # Filter for Calls
                        greeks = data.get("greeks", {})
                        implied_vol = greeks.get("implied_volatility")
                        
                        # Extracting parameters safely from the nested payload
                        if implied_vol and implied_vol > 0:
                            trade = data.get("latestTrade", {})
                            valid_contracts.append({
                                "symbol": symbol,
                                "iv": implied_vol,
                                "price": trade.get("p", 0)
                            })
                
                if not valid_contracts:
                    # Robust failover loop back to historical baseline if IV arrays are completely blank
                    sigma = 0.25
                    st.warning("Indicative option matrix values empty for this snapshot window. Defaulting to 25% asset baseline IV.")
                else:
                    # Use closest available profile
                    sigma = valid_contracts[0]["iv"]
                
                # Institutional floor proxy for interest yields (4.5%)
                r = 0.045
                
                # Compute estimated time duration (Hardcoded placeholder mapping to roughly 30 days out for execution test)
                T = 30 / 365.0
                
                delta = bs_call_delta(current_price, strike_input, r, sigma, T)
                
                # --- RENDER WEB UI ---
                st.markdown("---")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Asset Price (Delayed)", f"${current_price:.2f}")
                res_col2.metric("Alpaca Indicative IV", f"{sigma*100:.1f}%")
                res_col3.metric("Computed Call Delta", f"{delta:.4f}")
                
        except Exception as e:
            st.error(f"Data stream handshake error. Check endpoint routing parameters.")
