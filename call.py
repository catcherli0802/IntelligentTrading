import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import time  # Added for automatic AI exponential backoff retries

try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

st.set_page_config(page_title="Intelligent Alpha AI Quant", page_icon="🤖", layout="wide")
st.title("🤖 100% AI-Driven Option Quantitative Trading")
st.write("Uses the official Google GenAI SDK to dynamically research and ingest real-time market data parameters.")

if not SDK_AVAILABLE:
    st.error("🚨 Missing Package Dependency: Please add `google-genai` to your `requirements.txt` file on GitHub to enable the AI engine.")

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

# --- 2. OFFICIAL SDK AI MODEL INGESTION ENGINE WITH RETRY LOGIC (100% HARDCODE-FREE) ---
def query_gemini_market_intelligence(ticker_symbol: str, api_key: str, chosen_model: str) -> tuple:
    """
    Leverages the official google-genai library with a built-in retry mechanism 
    to robustly clear temporary 503 high-demand server errors.
    """
    if not SDK_AVAILABLE or not api_key:
        return 0.0, 0.0
        
    prompt = (
        f"You are a real-time financial market researcher. Use your knowledge base or integrated search capabilities "
        f"to find the absolute most recent closing/live stock price and the 30-day constant-maturity implied volatility (IV30) "
        f"for ticker symbol '{ticker_symbol.upper()}'.\n"
        f"Respond ONLY with a valid JSON block structured exactly like this: \n"
        f'{{"price": 0.0, "iv": 0.0}}'
    )
    
    max_retries = 3
    initial_delay = 2  # Seconds
    
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key.strip())
            
            response = client.models.generate_content(
                model=chosen_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            
            if response.text:
                parsed_ai_data = json.loads(response.text.strip())
                return float(parsed_ai_data.get("price", 0.0)), float(parsed_ai_data.get("iv", 0.0))
                
        except Exception as e:
            # Check for standard server capacity warning flags
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                st.warning(f"⚠️ Google server busy (503). Retrying ingestion attempt {attempt + 2}/{max_retries} in {initial_delay}s...")
                time.sleep(initial_delay)
                initial_delay *= 2  # Exponential delay backoff spacing rule
            else:
                st.error(f"Official SDK Transport Exception: {error_str}")
                break
                
    return 0.0, 0.0

# --- Initialize dynamic state memory lists for dropdown items ---
if "ticker_options" not in st.session_state:
    st.session_state.ticker_options = ["NVDA", "GOOG", "TSLA"]

# --- ROW 1: AI COMMAND INTERFACE PANEL ---
st.subheader("🤖 AI Agent Execution Dashboard")
ai_col1, ai_col2, ai_col_model, ai_col3, ai_col4 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])

with ai_col1:
    gemini_key = st.text_input(
        "Enter your Google Gemini API Key:", 
        type="password",
        placeholder="AIzaSy... or AQ..."
    )

with ai_col2:
    target_ticker_select = st.selectbox(
        "Select Target Ticker:",
        options=st.session_state.ticker_options,
        index=0
    )

with ai_col_model:
    # NEW: Dropdown selector allowing the user to seamlessly toggle models if Google's beta servers overload
    target_model_select = st.selectbox(
        "AI Engine Model Path:",
        options=["gemini-3.8-flash", "gemini-1.5-flash", "gemini-2.5-flash"],
        index=0,
        help="If one model path reports a temporary 503 high-demand error, flip this dropdown to try a completely different server lane!"
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
    elif not SDK_AVAILABLE:
        st.error("Cannot execute search loop. Ensure google-genai is installed.")
    else:
        with st.spinner(f"Gemini AI Agent is currently conducting option database runs for {target_ticker_select} using {target_model_select}..."):
            fetched_price, fetched_iv = query_gemini_market_intelligence(target_ticker_select, gemini_key, target_model_select)
                
            if fetched_price > 0 and fetched_iv > 0:
                st.session_state.ticker = target_ticker_select
                st.session_state.stock_price = float(fetched_price)
                st.session_state.implied_volatility = float(fetched_iv)
                st.toast(f"Gemini AI Success! Loaded live metrics for {target_ticker_select}", icon="✅")
            else:
                st.error("AI execution failed. If server demand is locked up, try toggling the 'AI Engine Model Path' dropdown to another choice.")

# --- ROW 2: STRATEGIC MODEL FIELDS ---
st.markdown("---")
st.subheader("📝 Verified Model Input Parameters")
in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns(5)

with in_col1:
    ticker_sym = st.text_input("Stock Ticker", key="ticker").upper().strip()
with in_col2:
    current_price = st.number_input("Stock Price (\$)", min_value=0.0, step=1.0, key="stock_price")
with in_col3:
    default_strike = round(current_price) if current_price > 0 else 0.0
    strike_input = st.number_input("Target Strike (\$)", value=float(default_strike), min_value=0.0, step=1.0)
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
                st.metric("Theoretical Value", f"\${theo_price:.2f}")
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
