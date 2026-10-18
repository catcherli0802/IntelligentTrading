import streamlit as st
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import requests
import re  

st.set_page_config(page_title="Intelligent Alpha AI Quant", page_icon="🤖", layout="wide")
st.title("🤖 AI-Driven Option Quantitative Trading")
st.write("Leverages an integrated AI agent workflow to dynamically extract, verify, and ingest real-time Implied Volatility (IV).")

# --- 1. CORE QUANTITATIVE MATH ENGINE ---
def calculate_greeks(S, K, r, sigma, T):
    """
    Computes analytical Black-Scholes formulas for a European Call Option.
    """
    if T <= 0 or sigma <= 0:
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

# --- 2. THE AI AGENT INTERNAL DATA TOOL ---
def ai_market_data_tool(ticker_symbol: str) -> str:
    """
    Tool execution layer for the AI Agent. Directly pulls JSON market parameters.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker_symbol.upper()}"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return json.dumps({"error": f"Ticker '{ticker_symbol}' not recognized by market endpoints."})
            
        data = response.json()
        result = data['optionChain']['result']
        if not result:
            return json.dumps({"error": f"No active options data found for ticker '{ticker_symbol}'."})
            
        quote_data = result[0].get('quote', {})
        live_price = float(quote_data.get('regularMarketPrice', 0))
        
        options_list = result[0].get('options', [])
        if not options_list or 'calls' not in options_list[0]:
            return json.dumps({"error": f"Options chain empty for ticker '{ticker_symbol}'."})
            
        calls = options_list[0]['calls']
        
        # Sample the exact At-The-Money option contract
        closest_option = min(calls, key=lambda x: abs(x['strike'] - live_price))
        live_iv = float(closest_option['impliedVolatility'] * 100.0)
        
        return json.dumps({
            "ticker": ticker_symbol.upper(),
            "stock_price": round(live_price, 2),
            "implied_volatility": round(live_iv, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return json.dumps({"error": f"Failed processing tool data: {str(e)}"})

# --- 3. COUPLING AN OPEN AI-ORCHESTRATION PIPELINE ---
def run_ai_agent_ingestion(prompt_input: str):
    """
    Simulates the cognitive workflow of an AI Agent:
    Uses strict regex and comprehensive stop-word validation to extract tickers cleanly.
    """
    # Comprehensive set of common words to block them from being treated as tickers
    stop_words = {
        "GET", "FETCH", "THE", "LIVE", "WHAT", "APP", "NOW", "AND", "FOR", "RIGHT", 
        "THIS", "DATA", "STOCK", "PRICE", "METRICS", "INFO", "SHOW", "CHECK", "RUN",
        "LOOP", "MARKET", "DYNAMIC", "CURRENT", "VAL", "CALL", "PUT", "OPTION", "GREEK"
    }
    
    words = re.findall(r'\b[A-Za-z]+\b', prompt_input.upper())
    
    # Strategy 1: Look for words matching common verified tech tickers immediately
    known_tickers = {"NVDA", "AAPL", "MSFT", "AMD", "TSLA", "GOOG", "META", "AMZN", "NFLX"}
    for word in words:
        if word in known_tickers:
            tool_output = ai_market_data_tool(word)
            return json.loads(tool_output)
            
    # Strategy 2: If no known tech ticker matches, pick the first 1-4 letter word not in stop_words
    target_ticker = "NVDA"  # Default safe backup
    for word in words:
        if 1 <= len(word) <= 4 and word not in stop_words:
            target_ticker = word
            break
            
    tool_output_json = ai_market_data_tool(target_ticker)
    parsed_data = json.loads(tool_output_json)
    return parsed_data

# --- ROW 1: AI COMMAND INTERFACE PANEL ---
st.subheader("🤖 AI Agent Execution Dashboard")
ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    user_ai_prompt = st.text_input(
        "Ask the AI Agent to fetch parameters (Natural Language):", 
        value="Get the live market price and dynamic IV metrics for NVDA right now"
    )
with ai_col2:
    st.write(" ")
    st.write(" ")
    trigger_agent = st.button("🚀 Run AI Ingestion Loop", use_container_width=True)

# Application Session State management to hold live state changes smoothly
if "ai_data" not in st.session_state:
    st.session_state.ai_data = {"ticker": "NVDA", "stock_price": 138.00, "implied_volatility": 35.7}

if trigger_agent:
    with st.spinner("AI Agent analyzing intent and running option chain tool execution loops..."):
        agent_result = run_ai_agent_ingestion(user_ai_prompt)
        if "error" not in agent_result:
            st.session_state.ai_data = agent_result
            st.toast(f"AI Success! Ingested live data streams for {agent_result['ticker']}", icon="✅")
        else:
            st.error(f"AI Agent Tool Breakdown: {agent_result['error']}")

# --- ROW 2: STRATEGIC MODEL FIELDS ---
st.markdown("---")
st.subheader("📝 Verified Model Input Parameters")
in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns(5)

with in_col1:
    ticker_sym = st.text_input("Stock Ticker", value=st.session_state.ai_data["ticker"]).upper().strip()
with in_col2:
    current_price = st.number_input("Stock Price ($)", value=float(st.session_state.ai_data["stock_price"]), min_value=0.01, step=1.0)
with in_col3:
    default_strike = round(current_price)
    strike_input = st.number_input("Target Strike ($)", value=float(default_strike), min_value=0.01, step=1.0)
with in_col4:
    iv_percent = st.number_input("Implied Volatility (σ %)", value=float(st.session_state.ai_data["implied_volatility"]), min_value=0.1, max_value=500.0, step=0.5)
with in_col5:
    maturity_input = st.text_input("Maturity (YYYY-MM-DD)", value="2026-11-20")

# Institutional risk-free bond yield baseline proxy configuration
r = 0.045

# --- ROW 3: DASHBOARD QUANT PERFORMANCE METRICS ---
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
