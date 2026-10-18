import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import requests

st.set_page_config(page_title="Intelligent Alpha AI Quant", page_icon="🤖", layout="wide")
st.title("🤖 100% AI-Driven Option Quantitative Trading")
st.write("Uses the Gemini API to dynamically research and ingest real-time market data parameters.")

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

# --- 2. LIVE AI MODEL INGESTION ENGINE (100% HARDCODE-FREE / ZERO FALLBACKS) ---
def query_gemini_market_intelligence(ticker_symbol: str, api_key: str) -> tuple:
    """
    Calls the Google Gemini API. Prioritizes Project-ID-Free standard Studio keys.
    Returns (0.0, 0.0) as a clean baseline if the API request triggers an error.
    """
    if not api_key:
        return 0.0, 0.0

    api_key_clean = api_key.strip()
    
    if api_key_clean.startswith("AQ."):
        url = "https://googleapis.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key_clean}"
        }
    else:
        url = f"https://googleapis.com{api_key_clean}"
        headers = {
            "Content-Type": "application/json"
        }

    prompt = (
        f"You are a real-time financial market researcher. Use your knowledge base or integrated search capabilities "
        f"to find the absolute most recent closing/live stock price and the 30-day constant-maturity implied volatility (IV30) "
        f"for ticker symbol '{ticker_symbol.upper()}'.\n"
        f"Respond ONLY with a valid JSON block structured exactly like this: \n"
        f'{{"price": 0.0, "iv": 0.0}}'
    )
    
    if api_key_clean.startswith("AQ."):
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {"responseMimeType": "application/json"}
        }
    else:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            
            if api_key_clean.startswith("AQ."):
                raw_text = res_json['predictions']['content']
            else:
                raw_text = res_json['candidates']['content']['parts']['text']
                
            parsed_ai_data = json.loads(raw_text.strip())
            return float(parsed_ai_data["price"]), float(parsed_ai_data["iv"])
        else:
            st.error(f"Gemini API returned status code {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Error handling AI response buffer: {str(e)}")
        
    # FIXED: Replaced hidden fallback matrix dictionaries entirely with neutral zero outputs
    return 0.0, 0.0

# --- Initialize dynamic state memory lists for dropdown items ---
if "ticker_options" not in st.session_state:
    st.session_state.ticker_options = ["NVDA", "GOOG", "TSLA"]

# --- ROW 1: AI COMMAND INTERFACE PANEL ---
st.subheader("🤖 AI Agent Execution Dashboard")
ai_col1, ai_col2, ai_col3, ai_col4 = st.columns([3, 2, 1.5, 1.5])

with ai_col1:
    gemini_key = st.text_input(
        "Enter your Google Gemini API Key:", 
        type="password",
        placeholder="AIzaSy... (Recommended) or AQ..."
    )
    st.caption("🔑 Clean architecture built for direct AI Studio keys.")

with ai_col2:
    target_ticker_select = st.selectbox(
        "Select Target Ticker:",
        options=st.session_state.ticker_options,
        index=0
    )

with ai_col3:
    new_custom_ticker = st.text_input("Add Custom Ticker:", placeholder="e.g. AMD, AAPL").upper().strip()
    click_add = st.button("➕ Add Ticker", use_container_width=True)
    
    if click_add and new_custom_ticker:
        if new_custom_ticker not in st.session_state.ticker_options:
            st.session_state.ticker_options.append(new_custom_ticker)
            st.rerun() 
        else:
            st.warning("Ticker already exists in choices!")

with ai_col4:
    st.write(" ")
    st.write(" ")
    trigger_agent = st.button("🚀 Run AI Ingestion Loop", use_container_width=True)

# Initialize fields completely pristine and blank to avoid hardcoded default visuals
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "stock_price" not in st.session_state:
    st.session_state.stock_price = 0.0
if "implied_volatility" not in st.session_state:
    st.session_state.implied_volatility = 0.0

# Process dynamic AI model responses instantly on execution click patterns
if trigger_agent:
    if not gemini_key:
        st.error("Please enter a valid Gemini API Key first.")
    else:
        with st.spinner(f"Gemini AI Agent is currently analyzing and researching options data for {target_ticker_select}..."):
            fetched_price, fetched_iv = query_gemini_market_intelligence(target_ticker_select, gemini_key)
                
            if fetched_price > 0 and fetched_iv > 0:
                st.session_state.ticker = target_ticker_select
                st.session_state.stock_price = float(fetched_price)
                st.session_state.implied_volatility = float(fetched_iv)
                st.toast(f"Gemini AI Success! Loaded live metrics for {target_ticker_select}", icon="✅")
            else:
                st.error("AI execution returned 0.0 metrics. Verify your key structure or switch to a standard AI Studio key.")

# --- ROW 2: STRATEGIC MODEL FIELDS ---
st.markdown("---")
st.subheader("📝 Verified Model Input Parameters")
in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns(5)

with in_col1:
    ticker_sym = st.text_input("Stock Ticker", key="ticker").upper().strip()
with in_col2:
    current_price = st.number_input("Stock Price ($)", min_value=0.0, step=1.0, key="stock_price")
with in_col3:
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
    st.info("💡 Enter your API Key, select or add an option ticker above, and click 'Run AI Ingestion Loop' to trigger the live Gemini AI engine.")
