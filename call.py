import streamlit as st
# Import your custom modules directly from call.py
from call import get_option_chain_data, analyze_with_gemini

st.title("Options Analytics POC")
ticker = st.text_input("Ticker", value="AAPL")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if st.button("Run Analysis") and api_key:
    # 1. Fetch data through backend logic
    calls, puts, spot = get_option_chain_data(ticker)
    
    # 2. Render tables in UI
    st.dataframe(calls)
    
    # 3. Fire Gemini analysis pipeline
    with st.spinner("Analyzing market structure..."):
        report = analyze_with_gemini(ticker, spot, calls, puts, api_key)
    st.markdown(report)
