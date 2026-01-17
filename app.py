
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from stock_utils import get_stock_data, preprocess_data
from model_utils import create_lstm_model, train_model, predict_future

# Set page config
st.set_page_config(layout="wide", page_title="NSE Stock Forecaster", page_icon="📈")


# Custom CSS for Light Theme
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #31333f;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(45deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .status-text {
        color: #00c853;
        font-weight: bold;
        font-size: 1.1em;
    }
    /* Metric Cards */
    .metric-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .metric-label {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .metric-value {
        color: #212529;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Enhanced Design
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("📊 Data Settings", expanded=True):
        ticker_input = st.text_input("Stock Ticker (NSE)", value="RELIANCE", help="Enter symbol like TCS, INFY. .NS added automatically.")
        years = st.slider("Historical Data (Years)", 1, 10, 5, help="More years = better training data but slower.")

    with st.expander("🤖 Model Settings", expanded=True):
        forecast_days = st.slider("Forecast Horizon (Days)", 5, 60, 10, help="How many days into the future to predict.")
        
    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)
    
    st.markdown("### ℹ️ About")
    st.info("Uses LSTM Neural Networks to predict short-term stock trends based on historical NSE data.")

# Main Area - Real-time Clock
from streamlit.components.v1 import html

# Header with Clock
c1, c2 = st.columns([3, 1])
with c1:
    st.title("📈 Indian NSE Stock Forecasting (v2.1)")

with c2:
    # Use an iframe component for the clock to ensure JS execution
    html("""
    <div style="text-align: right; font-family: monospace; color: #1f77b4; background: #e3f2fd; padding: 10px; border-radius: 8px;">
        <div id="clock" style="font-size: 1.2rem; font-weight: bold;">Loading...</div>
        <div id="date" style="font-size: 0.8rem;">Loading...</div>
    </div>
    <script>
    function updateTime() {
        const now = new Date();
        const options = { timeZone: 'Asia/Kolkata', hour12: false };
        const timeStr = now.toLocaleTimeString('en-US', options);
        const dateStr = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'Asia/Kolkata' });
        
        document.getElementById('clock').innerHTML = timeStr;
        document.getElementById('date').innerHTML = dateStr;
    }
    setInterval(updateTime, 1000);
    updateTime();
    </script>
    """, height=85)
    
st.markdown("Predict stock prices using **LSTM Recurrent Neural Networks**.")

if run_btn:
    status_text = st.markdown('<p class="status-text">### Status: ⏳ Fetching Data...</p>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    # 1. Fetch Data
    start_date = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    # yfinance end_date is exclusive, so add 1 day to include today
    end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    with st.spinner("Fetching data for " + ticker_input + "..."):
        data = get_stock_data(ticker_input, start_date, end_date)
        
    if data is None:
        status_text.markdown('<p class="status-text" style="color: #d32f2f;">### Status: ❌ Error Fetching Data. Check Ticker.</p>', unsafe_allow_html=True)
    else:
        progress_bar.progress(20)
        status_text.markdown('<p class="status-text" style="color: #2e7d32;">### Status: ✅ Data Fetched. Training Model...</p>', unsafe_allow_html=True)
        
        if 'latest_history' in locals() and not latest_history.empty:
            latest_date = latest_history.index[-1]
            if latest_date > data.index[-1]:
                 try:
                     new_row = latest_history.iloc[[-1]] 
                     common_cols = data.columns.intersection(new_row.columns)
                     if not common_cols.empty:
                        data = pd.concat([data, new_row[common_cols]])
                 except Exception as e:
                     print(f"Error merging data: {e}")

        # Helper to extract Series from potential DataFrame
        def get_close_series(df):
            if isinstance(df['Close'], pd.DataFrame):
                return df['Close'].iloc[:, 0]
            return df['Close']

        close_series = get_close_series(data)

        # Display Historical Data
        latest_date_display = data.index[-1].strftime('%d %b %Y')
        st.subheader(f"Historical Data for {ticker_input}.NS (Up to {latest_date_display})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=close_series, mode='lines', name='Close Price',
                                line=dict(color='#2196F3', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(33, 150, 243, 0.1)'))
        
        # Interactive Range Selector & Pointer Cursor Fix
        fig.update_xaxes(
            rangeslider_visible=False, # REMOVE the bottom slider which is confusing
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            showspikes=False # Disable spikes
        )
        
        fig.update_yaxes(showspikes=False)
        
        fig.layout.update(
            title_text=f'Time Series Data ({latest_date_display})', 
            template="plotly_white", 
            title_font=dict(color='#1f77b4'),
            font=dict(color='#212529'),
            hovermode="closest" # Switch to simple pointer (closest point) only
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Preprocess
        X_train, y_train, scaler = preprocess_data(data)
        
        if len(X_train) == 0:
             st.error("Not enough data to train model. Try increasing years of historical data.")
        else:
            progress_bar.progress(40)
            
            # 3. Create & Train Model
            model = create_lstm_model((X_train.shape[1], 1))
            
            with st.spinner("Training LSTM Model... This may take a moment."):
                # Reduced epochs for demo speed, increase for accuracy
                train_model(model, X_train, y_train, epochs=5, batch_size=32)
                
            progress_bar.progress(80)
            status_text.markdown('<p class="status-text" style="color: #1976d2;">### Status: 🤖 Model Trained. Forecasting...</p>', unsafe_allow_html=True)
            
            # 4. Predict
            last_sequence = data['Close'].values[-60:]
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            
            predictions = predict_future(model, last_sequence_scaled, scaler, days=forecast_days)
            
            progress_bar.progress(100)
            status_text.markdown('<p class="status-text">### Status: ✨ Process Complete!</p>', unsafe_allow_html=True)
            
            # Display Metrics
            try:
                # 1. Get latest close from the bulk data
                latest_close_series = data['Close']
                if isinstance(latest_close_series, pd.DataFrame):
                    latest_close_series = latest_close_series.iloc[:, 0]
                
                # 2. Try to fetch the absolute latest 'real-time'
                ticker_symbol = ticker_input if ticker_input.endswith('.NS') else f"{ticker_input}.NS"
                ticker_obj = yf.Ticker(ticker_symbol)
                
                # Method A: history(period='1mo')
                latest_history = ticker_obj.history(period='1mo') 
                
                if not latest_history.empty:
                     latest_price = float(latest_history['Close'].iloc[-1])
                     latest_date_val = latest_history.index[-1]
                     latest_date_str = latest_date_val.strftime('%d-%b-%Y')
                else:
                    # Method B: download
                     latest_data_dl = yf.download(ticker_symbol, period='1d')
                     if not latest_data_dl.empty:
                          latest_price = float(latest_data_dl['Close'].iloc[-1])
                          latest_date_str = latest_data_dl.index[-1].strftime('%d-%b-%Y')
                     else:
                        latest_price = float(latest_close_series.iloc[-1])
                        latest_date_str = data.index[-1].strftime('%d-%b-%Y')
                    
            except Exception as e:
                st.error(f"Error extracting price: {e}")
                latest_price = 0.0
                latest_date_str = "N/A"

            next_price = float(predictions[0][0])
            price_change = next_price - latest_price
            img_arrow = "⬆️" if price_change > 0 else "⬇️"
            trend_color = "#00c853" if price_change > 0 else "#d32f2f"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 class="metric-label">Close Price ({latest_date_str})</h3>
                    <h1 class="metric-value">₹{latest_price:.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 class="metric-label">Next Day Forecast</h3>
                    <h1 class="metric-value" style="color: {trend_color};">{img_arrow} ₹{next_price:.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")

            # Display Forecast
            st.subheader(f"Forecast for next {forecast_days} days")
            
            last_date_obj = latest_history.index[-1].date() if 'latest_history' in locals() and not latest_history.empty else data.index[-1].date()
            future_dates = [last_date_obj + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(x=data.index, y=close_series, mode='lines', name='History',
                                    line=dict(color='#90a4ae', width=1.5))) 
            fig_pred.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines+markers', name='Forecast',
                                    line=dict(color='#2196F3', width=3, shape='spline'), 
                                    marker=dict(size=5, color='#1976d2', line=dict(width=1, color='white'))))
            
            fig_pred.update_xaxes(showspikes=False, rangeslider_visible=False) # Removal here too
            fig_pred.update_yaxes(showspikes=False)

            fig_pred.layout.update(
                title_text='Stock Price Forecasting', 
                template="plotly_white",
                font=dict(color='#212529'),
                hovermode="closest"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.warning("Note: This is a simplified demo. Actual stock market prediction is extremely complex and risky. Do not use this for real financial decisions.")
