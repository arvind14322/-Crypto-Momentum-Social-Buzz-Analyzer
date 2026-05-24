import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_fetcher import get_combined_dataset
from analysis import (
    compute_cross_correlation, 
    run_granger_causality, 
    fit_garch_model, 
    run_event_study
)

# Set page config
st.set_page_config(
    page_title="Crypto Buzz & Momentum Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">Crypto Momentum & Social Buzz Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Examine lead-lag relationships, causality, and event effects between crypto prices and social buzz.</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("Configuration")

# Coin Selection
coin_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Cardano": "cardano",
    "Dogecoin": "dogecoin"
}
selected_coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(coin_options.keys()))
coin_id = coin_options[selected_coin_name]

# Data source toggle
data_source = st.sidebar.radio("Data Source", ["Simulated (Ideal for testing relations)", "Real Price + Simulated Buzz"])
use_real = data_source == "Real Price + Simulated Buzz"

# Simulation Mode (Ground Truth)
sim_mode = st.sidebar.selectbox(
    "Ground Truth Causal Relationship",
    ["social_leads", "price_leads", "no_causality"],
    format_func=lambda x: {
        "social_leads": "Social Buzz Leads Price (24-72h)",
        "price_leads": "Price Jumps Lead Social Buzz",
        "no_causality": "Independent (No Causality)"
    }[x]
)

days = st.sidebar.slider("Historical Window (Days)", min_value=30, max_value=365, value=120)
noise_level = st.sidebar.slider("Social Buzz Noise Level", min_value=0.05, max_value=0.5, value=0.15, step=0.05)

# Fetch data
with st.spinner("Fetching and preparing dataset..."):
    df, price_is_simulated = get_combined_dataset(
        coin_id=coin_id,
        days=days,
        causality_mode=sim_mode,
        noise_level=noise_level,
        use_real_price=use_real
    )

# Alert user about price source
if price_is_simulated:
    st.info("⚠️ CoinGecko API unavailable or simulator selected. Using generated synthetic price path.")
else:
    st.success(f"📈 Loaded real-time CoinGecko price data for {selected_coin_name}.")

# -----------------
# 1. METRIC CARDS
# -----------------
col1, col2, col3, col4 = st.columns(4)

# Calculate stats
current_price = df["price"].iloc[-1]
price_pct_change = ((df["price"].iloc[-1] - df["price"].iloc[0]) / df["price"].iloc[0]) * 100
avg_mentions = df["social_mentions"].mean()
max_mentions = df["social_mentions"].max()

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Price ({selected_coin_name})</div>
        <div class="metric-value">${current_price:,.2f}</div>
        <div class="metric-delta" style="color: {'#10b981' if price_pct_change >= 0 else '#ef4444'};">
            {'+' if price_pct_change >= 0 else ''}{price_pct_change:.2f}% (period)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Daily Social Mentions</div>
        <div class="metric-value">{int(avg_mentions):,}</div>
        <div class="metric-delta" style="color: #94a3b8;">
            Peak mentions: {int(max_mentions):,}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Granger results for col3 & col4
granger_res = run_granger_causality(df, max_lag=3)

with col3:
    if "error" not in granger_res:
        p_social_leads = granger_res["social_causes_price"].get(2, 1.0)
        is_sig = p_social_leads < 0.05
        status_text = "Significant Causality" if is_sig else "Not Significant"
        status_color = "#10b981" if is_sig else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Social Granger-Causes Price (Lag 2)</div>
            <div class="metric-value">{p_social_leads:.3f}</div>
            <div class="metric-delta" style="color: {status_color}; font-weight: bold;">
                {status_text} (p-val)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Social Granger-Causes Price</div>
            <div class="metric-value">N/A</div>
            <div class="metric-delta" style="color: #ef4444;">Insufficient data</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    if "error" not in granger_res:
        p_price_leads = granger_res["price_causes_social"].get(2, 1.0)
        is_sig = p_price_leads < 0.05
        status_text = "Significant Causality" if is_sig else "Not Significant"
        status_color = "#10b981" if is_sig else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Price Granger-Causes Social (Lag 2)</div>
            <div class="metric-value">{p_price_leads:.3f}</div>
            <div class="metric-delta" style="color: {status_color}; font-weight: bold;">
                {status_text} (p-val)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Price Granger-Causes Social</div>
            <div class="metric-value">N/A</div>
            <div class="metric-delta" style="color: #ef4444;">Insufficient data</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------
# 2. TIME-SERIES VISUALIZATION
# -----------------
st.subheader("Price vs Social Mentions Time-series")

fig = go.Figure()

# Add Price trace
fig.add_trace(go.Scatter(
    x=df.index, y=df["price"],
    name="Price (USD)",
    line=dict(color="#3b82f6", width=2),
    yaxis="y1"
))

# Add Social Mentions trace
fig.add_trace(go.Bar(
    x=df.index, y=df["social_mentions"],
    name="Social Mentions",
    marker_color="rgba(168, 85, 247, 0.4)",
    yaxis="y2"
))

# Layout with dual Y-axis
fig.update_layout(
    yaxis=dict(
        title=dict(text="Price (USD)", font=dict(color="#3b82f6")),
        tickfont=dict(color="#3b82f6"),
    ),
    yaxis2=dict(
        title=dict(text="Social Mentions", font=dict(color="#a855f7")),
        tickfont=dict(color="#a855f7"),
        anchor="x",
        overlaying="y",
        side="right"
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=40, t=20, b=40),
    height=450,
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------
# 3. STATISTICAL ANALYSIS PANELS
# -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Cross-Correlation", 
    "🔮 Granger Causality", 
    "📈 Volatility (GARCH)", 
    "⚡ Event Study"
])

with tab1:
    st.subheader("Cross-Correlation Analysis")
    st.write("Does social buzz spike *before* (leads) or *after* (lags) price returns?")
    
    max_lag_val = st.slider("Max Correlation Lag (Days)", 3, 15, 10)
    cc_df = compute_cross_correlation(df, max_lag=max_lag_val)
    
    # Highlight max correlation
    optimal_lag_idx = cc_df["correlation"].abs().idxmax()
    opt_lag = cc_df.loc[optimal_lag_idx, "lag"]
    opt_corr = cc_df.loc[optimal_lag_idx, "correlation"]
    
    # Plot cross correlation
    cc_fig = px.bar(
        cc_df, x="lag", y="correlation",
        title="Cross-Correlation at Different Lags",
        labels={"lag": "Lag (Days)", "correlation": "Correlation Coefficient"},
        color="correlation",
        color_continuous_scale=px.colors.sequential.Viridis,
        template="plotly_dark"
    )
    cc_fig.update_layout(height=400, coloraxis_showscale=False)
    
    st.plotly_chart(cc_fig, use_container_width=True)
    
    if opt_lag > 0:
        st.markdown(f"**Conclusion:** Social buzz **leads** price returns by **{opt_lag} days** with a correlation of **{opt_corr:.3f}**.")
    elif opt_lag < 0:
        st.markdown(f"**Conclusion:** Price returns **lead** social buzz by **{abs(opt_lag)} days** with a correlation of **{opt_corr:.3f}**.")
    else:
        st.markdown(f"**Conclusion:** Contemporary correlation is strongest (correlation of **{opt_corr:.3f}**).")

with tab2:
    st.subheader("Granger Causality Test")
    st.write("Granger causality tests if historical values of variable X contain info that helps predict variable Y beyond historical values of Y alone.")
    
    if "error" not in granger_res:
        g_df = pd.DataFrame({
            "Lag (Days)": list(granger_res["social_causes_price"].keys()),
            "Social -> Price (p-val)": [f"{p:.4f}" for p in granger_res["social_causes_price"].values()],
            "Price -> Social (p-val)": [f"{p:.4f}" for p in granger_res["price_causes_social"].values()]
        })
        
        st.dataframe(g_df, use_container_width=True)
        st.markdown("""
        *A p-value < 0.05 indicates statistical significance (we reject the null hypothesis that variable X does NOT Granger-cause Y).*
        """)
        
        # Check actual results
        soc_causes_p = granger_res["social_causes_price"].get(2, 1.0)
        prc_causes_p = granger_res["price_causes_social"].get(2, 1.0)
        
        if soc_causes_p < 0.05 and prc_causes_p >= 0.05:
            st.success("🎯 **Result:** Social buzz Granger-causes price returns! Social spikes contain predictive value.")
        elif prc_causes_p < 0.05 and soc_causes_p >= 0.05:
            st.warning("🔄 **Result:** Price returns Granger-cause social mentions! Social buzz is likely a reaction to price jumps.")
        elif soc_causes_p < 0.05 and prc_causes_p < 0.05:
            st.info("🤝 **Result:** Feedback loop detected! Price returns and social buzz Granger-cause each other.")
        else:
            st.info("💤 **Result:** No significant causal relation detected at 2-day lag threshold.")
    else:
        st.error(f"Error executing Granger causality test: {granger_res.get('error')}")

with tab3:
    st.subheader("GARCH Volatility Modeling")
    st.write("Fitting a GARCH(1,1) model to estimate the asset's time-varying conditional volatility.")
    
    vol_series, model_info = fit_garch_model(df)
    
    if model_info.get("converged", False):
        st.success("GARCH(1,1) model fitted successfully!")
        
        # Plot volatility vs returns
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=df.index[1:], y=df["price"].pct_change().dropna() * 100,
            name="Returns (%)",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
        ))
        vol_fig.add_trace(go.Scatter(
            x=vol_series.index, y=vol_series * 100,
            name="GARCH Volatility (%)",
            line=dict(color="#f59e0b", width=2),
        ))
        vol_fig.update_layout(
            template="plotly_dark",
            height=400,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(vol_fig, use_container_width=True)
        
        # Parameter estimation table
        params = model_info["params"]
        param_df = pd.DataFrame({
            "Parameter": list(params.keys()),
            "Estimated Value": list(params.values())
        })
        st.write("Model Parameters:")
        st.dataframe(param_df, use_container_width=True)
    else:
        st.warning(f"GARCH failed to converge (falling back to simple rolling volatility). Error: {model_info.get('error')}")
        
        # Plot fallback
        vol_fig = px.line(
            x=vol_series.index, y=vol_series * 100,
            title="Rolling Historical Volatility (%)",
            template="plotly_dark"
        )
        st.plotly_chart(vol_fig, use_container_width=True)

with tab4:
    st.subheader("Event Study: Impact of Social Spike Events")
    st.write("Examines price behavior in a window around large social mention spikes (t=0).")
    
    pre_window = st.slider("Pre-event window (days)", 1, 5, 3)
    post_window = st.slider("Post-event window (days)", 2, 10, 5)
    
    es_df = run_event_study(df, window_pre=pre_window, window_post=post_window)
    
    if es_df["cum_return"].abs().sum() > 0:
        es_fig = go.Figure()
        
        # Abnormal return bar chart
        es_fig.add_trace(go.Bar(
            x=es_df["day"], y=es_df["mean_return"] * 100,
            name="Daily Mean Return (%)",
            marker_color="#10b981"
        ))
        
        # Cumulative return line chart
        es_fig.add_trace(go.Scatter(
            x=es_df["day"], y=es_df["cum_return"] * 100,
            name="Cumulative Return (%)",
            line=dict(color="#ef4444", width=3)
        ))
        
        es_fig.add_trace(go.Scatter(
            x=[0, 0], y=[es_df["cum_return"].min()*110, es_df["cum_return"].max()*110],
            mode="lines",
            line=dict(color="white", width=1, dash="dash"),
            name="Event Day (t=0)"
        ))
        
        es_fig.update_layout(
            template="plotly_dark",
            height=400,
            title="Mean and Cumulative Returns around Event Day (t=0)",
            xaxis=dict(title="Days relative to Event"),
            yaxis=dict(title="Return (%)"),
            hovermode="x unified"
        )
        
        st.plotly_chart(es_fig, use_container_width=True)
    else:
        st.info("No social spike events detected in the selected data range.")
