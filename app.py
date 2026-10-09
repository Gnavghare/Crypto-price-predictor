import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

# Page config - must be first
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern attractive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .btc-card { border-left: 4px solid #f7931a; }
    .eth-card { border-left: 4px solid #627eea; }
    .bnb-card { border-left: 4px solid #f3ba2f; }
    
    .prediction-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7931a, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    div[data-testid="stSidebar"] .stNumberInput label,
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    .feature-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 0.2rem;
    }
    
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.6rem; }
        .prediction-value { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('selected_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, selected_features = load_artifacts()

# Fetch live crypto data
@st.cache_data(ttl=60)
def get_live_prices():
    tickers = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'BNB': 'BNB-USD',
        'USDT': 'USDT-USD'
    }
    data = {}
    for name, ticker in tickers.items():
        t = yf.Ticker(ticker)
        hist = t.history(period='1d')
        if not hist.empty:
            data[name] = {
                'price': float(hist['Close'].iloc[-1]),
                'volume': float(hist['Volume'].iloc[-1]),
                'change': float(hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) if len(hist) > 0 else 0,
                'change_pct': float((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1] * 100) if len(hist) > 0 else 0
            }
        else:
            data[name] = {'price': 0, 'volume': 0, 'change': 0, 'change_pct': 0}
    return data

@st.cache_data(ttl=300)
def get_historical_data(period='6mo'):
    btc = yf.Ticker('BTC-USD').history(period=period)[['Close', 'Volume']].rename(columns={'Close': 'BTC', 'Volume': 'BTC_Vol'})
    eth = yf.Ticker('ETH-USD').history(period=period)[['Close', 'Volume']].rename(columns={'Close': 'ETH', 'Volume': 'ETH_Vol'})
    bnb = yf.Ticker('BNB-USD').history(period=period)[['Close', 'Volume']].rename(columns={'Close': 'BNB', 'Volume': 'BNB_Vol'})
    usdt = yf.Ticker('USDT-USD').history(period=period)[['Close', 'Volume']].rename(columns={'Close': 'USDT', 'Volume': 'USDT_Vol'})
    df = btc.join(eth).join(bnb).join(usdt).dropna()
    return df

def format_price(price):
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

def format_volume(vol):
    if vol >= 1e9:
        return f"${vol/1e9:.2f}B"
    elif vol >= 1e6:
        return f"${vol/1e6:.2f}M"
    else:
        return f"${vol:,.0f}"

# Header
st.markdown("""
<div class="main-header">
    <h1>₿ Crypto Price Predictor</h1>
    <p>AI-Powered Bitcoin Close Price Prediction using Random Forest • Live Market Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Live prices section
live = get_live_prices()

st.markdown("### 📊 Live Market Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    change_color = "#00c853" if live['BTC']['change'] >= 0 else "#ff1744"
    st.markdown(f"""
    <div class="metric-card btc-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">₿</span>
            <span style="font-weight:600;color:#f7931a;">Bitcoin</span>
        </div>
        <div style="font-size:1.5rem;font-weight:700;color:white;">{format_price(live['BTC']['price'])}</div>
        <div style="color:{change_color};font-size:0.9rem;margin-top:4px;">
            {'▲' if live['BTC']['change'] >= 0 else '▼'} {live['BTC']['change_pct']:+.2f}%
        </div>
        <div style="color:#a0aec0;font-size:0.8rem;margin-top:6px;">Vol: {format_volume(live['BTC']['volume'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    change_color = "#00c853" if live['ETH']['change'] >= 0 else "#ff1744"
    st.markdown(f"""
    <div class="metric-card eth-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">Ξ</span>
            <span style="font-weight:600;color:#627eea;">Ethereum</span>
        </div>
        <div style="font-size:1.5rem;font-weight:700;color:white;">{format_price(live['ETH']['price'])}</div>
        <div style="color:{change_color};font-size:0.9rem;margin-top:4px;">
            {'▲' if live['ETH']['change'] >= 0 else '▼'} {live['ETH']['change_pct']:+.2f}%
        </div>
        <div style="color:#a0aec0;font-size:0.8rem;margin-top:6px;">Vol: {format_volume(live['ETH']['volume'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    change_color = "#00c853" if live['BNB']['change'] >= 0 else "#ff1744"
    st.markdown(f"""
    <div class="metric-card bnb-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">🔶</span>
            <span style="font-weight:600;color:#f3ba2f;">BNB</span>
        </div>
        <div style="font-size:1.5rem;font-weight:700;color:white;">{format_price(live['BNB']['price'])}</div>
        <div style="color:{change_color};font-size:0.9rem;margin-top:4px;">
            {'▲' if live['BNB']['change'] >= 0 else '▼'} {live['BNB']['change_pct']:+.2f}%
        </div>
        <div style="color:#a0aec0;font-size:0.8rem;margin-top:6px;">Vol: {format_volume(live['BNB']['volume'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    change_color = "#00c853" if live['USDT']['change'] >= 0 else "#ff1744"
    st.markdown(f"""
    <div class="metric-card" style="border-left:4px solid #26a17b;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">💵</span>
            <span style="font-weight:600;color:#26a17b;">USDT</span>
        </div>
        <div style="font-size:1.5rem;font-weight:700;color:white;">{format_price(live['USDT']['price'])}</div>
        <div style="color:{change_color};font-size:0.9rem;margin-top:4px;">
            {'▲' if live['USDT']['change'] >= 0 else '▼'} {live['USDT']['change_pct']:+.4f}%
        </div>
        <div style="color:#a0aec0;font-size:0.8rem;margin-top:6px;">Vol: {format_volume(live['USDT']['volume'])}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Prediction and Charts
tab1, tab2, tab3 = st.tabs(["🔮 BTC Price Prediction", "📈 Price Charts", "ℹ️ About Model"])

with tab1:
    st.markdown("#### Predict Bitcoin Closing Price")
    st.caption(f"Model uses top 4 features selected by SelectKBest: **{', '.join(selected_features)}**")
    
    # Sidebar inputs (also available in main for mobile)
    with st.sidebar:
        st.markdown("## 🎛️ Input Features")
        st.markdown("---")
        
        st.markdown("**Ethereum (ETH)**")
        eth_close = st.number_input(
            "ETH Close Price ($)",
            min_value=0.0,
            value=float(round(live['ETH']['price'], 2)),
            format="%.2f",
            help="Current or expected ETH closing price"
        )
        eth_volume = st.number_input(
            "ETH Volume",
            min_value=0.0,
            value=float(round(live['ETH']['volume'], 0)),
            format="%.0f",
            help="ETH 24h trading volume"
        )
        
        st.markdown("**Tether (USDT)**")
        usdt_volume = st.number_input(
            "USDT Volume",
            min_value=0.0,
            value=float(round(live['USDT']['volume'], 0)),
            format="%.0f",
            help="USDT 24h trading volume"
        )
        
        st.markdown("**BNB**")
        bnb_close = st.number_input(
            "BNB Close Price ($)",
            min_value=0.0,
            value=float(round(live['BNB']['price'], 2)),
            format="%.2f",
            help="Current or expected BNB closing price"
        )
        
        st.markdown("---")
        st.markdown("💡 *Values pre-filled with live market data. Adjust as needed.*")
    
    # Main area prediction
    pred_col1, pred_col2 = st.columns([1.2, 1])
    
    with pred_col1:
        st.markdown("##### Feature Summary")
        summary_df = pd.DataFrame({
            'Feature': selected_features,
            'Value': [eth_close, eth_volume, usdt_volume, bnb_close],
            'Live Reference': [
                format_price(live['ETH']['price']),
                format_volume(live['ETH']['volume']),
                format_volume(live['USDT']['volume']),
                format_price(live['BNB']['price'])
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        if st.button("🚀 Predict BTC Close Price", use_container_width=True):
            # Prepare input in correct order matching selected_features
            input_dict = {
                'Close (ETH)': eth_close,
                'Volume (ETH)': eth_volume,
                'Volume (USDT)': usdt_volume,
                'Close (BNB)': bnb_close
            }
            input_df = pd.DataFrame([[input_dict[f] for f in selected_features]], columns=selected_features)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            st.session_state['prediction'] = prediction
            st.session_state['predicted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with pred_col2:
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            live_btc = live['BTC']['price']
            diff = pred - live_btc
            diff_pct = (diff / live_btc) * 100 if live_btc else 0
            
            st.markdown(f"""
            <div class="prediction-box">
                <div style="color:#a0aec0;font-size:0.95rem;">Predicted BTC Close</div>
                <div class="prediction-value">{format_price(pred)}</div>
                <div style="color:#cbd5e0;font-size:0.9rem;margin-top:0.5rem;">
                    vs Live: {format_price(live_btc)}
                </div>
                <div style="color:{'#00c853' if diff >= 0 else '#ff1744'};font-size:1.1rem;font-weight:600;margin-top:0.4rem;">
                    {'▲' if diff >= 0 else '▼'} {diff:+,.2f} ({diff_pct:+.2f}%)
                </div>
                <div style="color:#718096;font-size:0.75rem;margin-top:1rem;">
                    Predicted at {st.session_state.get('predicted_at', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box">
                <div style="color:#a0aec0;font-size:1rem;padding:2rem 0;">
                    👈 Adjust features in the sidebar<br>and click <b>Predict</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### Historical Price Trends")
    
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    hist = get_historical_data(period)
    
    # Price chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Closing Prices", "Trading Volume (BTC)")
    )
    
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['BTC'], name='BTC',
        line=dict(color='#f7931a', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['ETH'], name='ETH',
        line=dict(color='#627eea', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['BNB'], name='BNB',
        line=dict(color='#f3ba2f', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=hist.index, y=hist['BTC_Vol'], name='BTC Volume',
        marker_color='rgba(247, 147, 26, 0.5)'
    ), row=2, col=1)
    
    fig.update_layout(
        height=550,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap style
    st.markdown("##### Price Correlation")
    corr = hist[['BTC', 'ETH', 'BNB']].corr()
    fig_corr = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='Correlation between BTC, ETH & BNB'
    )
    fig_corr.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("""
    ### About This Model
    
    This application predicts **Bitcoin (BTC) closing price** using a **Random Forest Regressor** trained on 5 years of historical data from Yahoo Finance.
    
    #### Feature Selection
    - Original features: Close & Volume of BTC, ETH, USDT, BNB
    - Target: `Close (BTC)`
    - **SelectKBest** (f_regression) selected the top 4 most predictive features:
    """)
    
    for i, f in enumerate(selected_features, 1):
        st.markdown(f"  {i}. `{f}`")
    
    st.markdown("""
    #### Pipeline
    1. Data collected via `yfinance` (5-year history)
    2. Feature selection with SelectKBest (k=4)
    3. MinMaxScaler normalization
    4. Random Forest (100 estimators)
    
    #### Notes
    - Predictions are based on relationships learned from historical co-movements.
    - Crypto markets are highly volatile — this is **not financial advice**.
    - Model R² on held-out test data ≈ **0.97** (when trained on the same period).
    
    #### Tech Stack
    - **Frontend**: Streamlit + custom CSS + Plotly
    - **ML**: scikit-learn (RandomForest, SelectKBest, MinMaxScaler)
    - **Data**: yfinance
    """)

# Footer
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • Data from Yahoo Finance • Not financial advice
</div>
""", unsafe_allow_html=True)