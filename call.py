import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import requests

st.set_page_config(page_title="Intelligent Alpha AI Quant", page_icon="🤖", layout="wide")
st.title("🤖 AI-Driven Option Quantitative Trading")
st.write("Leverages an integrated AI workflow to dynamically ingest verified market parameters.")

# --- 1. CORE QUANTITATIVE MATH ENGINE ---
def calculate_greeks(S, K, r, sigma, T):
    """
    Computes analytical Black-Scholes formulas for a European Call Option.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000
        
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
    delta = cdf_d1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0
    theta_term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    theta_term2 = -r * K * np.exp(-r * T) * cdf_d2
    theta = (theta_term1 + theta_term2) / 365.0
    rho = (K * T * np.exp(-r * T) * cdf_d2) / 100.0
    
    return price, delta, gamma, vega, theta, rho

# --- 2. LIVE APIS CONNECTED DATA FEED ENGINE (NO HARDCODED VALUES) ---
def ai_live_market_tool(ticker_symbol: str) -> tuple:
    """
    Directly queries the live public options chain endpoint. 
    Dynamically extracts the spot price and samples the current ATM option implied volatility.
    """
    clean_ticker = str(ticker_symbol).upper().strip().replace("/", "").replace(" ", "")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        url = f"https://yahoo.com{clean_ticker}"
        response = requests.get(url, headers=headers, timeout=6)
        
        if response.status_code == 200:
            data = response.json()
            result_list = data.get('optionChain', {}).get('result', [])
            
            if result_list:
                result = result_list[0]
                
                # A. Dynamically parse active regular market price
                quote_data = result.get('quote', {})
                live_price = float(quote_data.get('regularMarketPrice', quote_data.get('previousClose', 0.0)))
                
                # B. Dynamically extract live Implied Volatility from the Call options arrays
                options_data = result.get('options', [])
                if options_data and len(options_data) > 0:
                    calls_list = options_data[0].get('calls', [])
                    
                    if calls_list:
                        # Locate the option contract string that rests directly At-The-Money
                        closest_option = min(calls_list, key=lambda x: abs(x.get('strike', 0) - live_price))
                        live_iv = float(closest_option.get('impliedVolatility', 0.0) * 100.0)
                        
                        return live_price, live_iv
                
                return live_price, 0.0
    except Exception:
        pass
    return 0.0, 0.0

# --- ROW 1: AI COMMAND INTERFACE PANEL ---
st.subheader("🤖 AI Ingestion Selector Panel")
ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    target_ticker_select = st.selectbox(
        "Select Target Ticker for real-time AI ingestion:",
        options=["NVDA", "GOOG", "TSLA"],
        index=0
    )
with ai_col2:
    st.write(" ")
    st.write(" ")
    trigger_agent = st.button("🚀 Run AI Ingestion Loop", use_container_width=True)

# FIXED: Initialized session state keys with blank/neutral parameters instead of hardcoded numbers
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "stock_price" not in st.session_state:
    st.session_state.stock_price = 0.0
if "implied_volatility" not in st.session_state:
    st.session_state.implied_volatility = 0.0

# Process data sync actions instantly using exact real-time pricing definitions
if trigger_agent:
    with st.spinner("AI parsing engine querying dynamic live-stream parameters..."):
        fetched_price, fetched_iv = ai_live_market_tool(target_ticker_select)
            
        if fetched_price > 0:
            st.session_state.ticker = target_ticker_select
            st.session_state.stock_price = float(fetched_price)
            st.session_state.implied_volatility = float(fetched_iv)
            st.toast(f"AI Success! Ingested live web parameters for {target_ticker_select}", icon="✅")
        else:
            st.error("Could not fetch real-time data. Check market connectivity.")

# --- ROW 2: STRATEGIC MODEL FIELDS ---
st.markdown("---")
st.subheader("📝 Verified Model Input Parameters")
in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns(5)

with in_col1:
    ticker_sym = st.text_input("Stock Ticker", key="ticker").upper().strip()
with in_col2:
    current_price = st.number_input("Stock Price ($)", min_value=0.0, step=1.0, key="stock_price")
with in_col3:
    # Safely compute the default strike step based on the loaded session parameters
    default_strike = round(current_price) if current_price > 0 else 0.0
    strike_input = st.number_input("Target Strike ($)", value=float(default_strike), min_value=0.0, step=1.0)
with in_col4:
    iv_percent = st.number_input("Implied Volatility (σ %)", min_value=0.0, max_value=500.0, step=0.5, key="implied_volatility")
with in_col5:
    maturity_input = st.text_input("Maturity (YYYY-MM-DD)", value="2026-11-20")

# Institutional risk-free bond yield baseline proxy configuration
r = 0.045

# --- ROW 3: DASHBOARD QUANT PERFORMANCE METRICS ---
if current_price > 0 and iv_percent > 0:
    try:
        today = datetime.now().date()
        target_expiry = datetime.strptime(maturity_input.strip(), "%Y-%m-%d").date()
        days_to_maturity = (target_expiry - today).days
        T = days_to_maturity / 365.0
        
        if days_to_maturity <= 0:
            st.error("Maturity date must be a date in the future.")
        else:
            sigma = iv_percent / 100.0
            theo_price, delta, gamma, vega, theta, rho = calculate_greeks(current_price, strike_input, r, sigma, T)
            
            st.markdown("---")
            st.subheader(f"📊 Quantitative Dashboard Results ({ticker_sym}) — {days_to_maturity} Days to Expiry")
            res_col0, res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(6)
            
            with res_col0:
                st.metric("Theoretical Value", f"${theo_price:.2f}")
            with res_col1:
                st.metric("Delta (Δ)", f"{delta * 100:.2f}%")
            with res_col2:
                st.metric("Gamma (Γ)", f"{gamma:.4f}")
            with res_col3:
                st.metric("Vega (ν)", f"{vega:.4f}")
            with res_col4:
                st.metric("Theta (θ)", f"{theta:.4f}")
            with res_col5:
                st.metric("Rho (ρ)", f"{rho:.4f}")
                
    except Exception as err:
        st.error(f"Mathematical execution exception: {str(err)}")
else:
    st.info("💡 Select an option ticker above and click 'Run AI Ingestion Loop' to view real-time Black-Scholes metrics.")
