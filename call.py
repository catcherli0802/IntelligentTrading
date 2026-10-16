import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

st.title("🧮 Option Delta Calculator")
st.write("Fetches live underlying prices and implied volatilities to calculate Black-Scholes Call Delta.")

col1, col2, col3 = st.columns(3)
with col1:
    ticker_sym = st.text_input("Stock Ticker", value="AAPL").upper().strip()
with col2:
    strike_price = st.number_input("Strike Price ($)", value=230.0, step=5.0)
with col3:
    maturity_str = st.text_input("Maturity Date (YYYY-MM-DD)", value="2027-01-15")

if st.button("Calculate Delta"):
    try:
        # Fetch underlying data
        stock = yf.Ticker(ticker_sym)
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        
        # Risk-free rate (3-Month Treasury Yield Proxy)
        tnx = yf.Ticker("^IRX")
        r = (tnx.history(period="1d")["Close"].iloc[-1]) / 100 
        
        # Time to expiration
        today = datetime.now().date()
        maturity_date = datetime.strptime(maturity_str, "%Y-%m-%d").date()
        days_to_maturity = (maturity_date - today).days
        
        if days_to_maturity <= 0:
            st.error("Please pick a future maturity date.")
        else:
            T = days_to_maturity / 365.0
            
            # Extract Implied Volatility from option chain
            opt_chain = stock.option_chain(maturity_str)
            calls = opt_chain.calls
            
            # Find closest strike mapping
            matched_option = calls[np.isclose(calls['strike'], strike_price, atol=5.0)]
            
            if matched_option.empty:
                st.warning("Exact strike not found. Defaulting to standard 25% atmospheric IV.")
                sigma = 0.25
            else:
                sigma = matched_option['impliedVolatility'].iloc[0]
            
            # Black-Scholes d1 calculation
            d1 = (np.log(current_price / strike_price) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            call_delta = norm.cdf(d1)
            
            # Display Dashboard
            st.markdown("---")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Stock Price", f"${current_price:.2f}")
            res_col2.metric("Market IV", f"{sigma*100:.1f}%")
            res_col3.metric("Days to Expiry", f"{days_to_maturity}")
            res_col4.metric("Call Delta (Δ)", f"{call_delta:.4f}")
            
    except Exception as e:
        st.error("Error fetching data. Check ticker validation or options availability for that date.")
