import pandas as pd
import numpy as np
import requests
import datetime
from typing import Tuple, Optional

def fetch_coingecko_data(coin_id: str = "bitcoin", days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetches daily price and volume data from CoinGecko.
    Returns a DataFrame with columns: ['price', 'volume'], indexed by datetime.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
            df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            
            df = pd.merge(df_prices, df_volumes, on="timestamp")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            df.drop(columns=["timestamp"], inplace=True)
            return df
    except Exception as e:
        print(f"CoinGecko fetch failed: {e}")
    return None

def generate_synthetic_data(
    days: int = 90, 
    causality_mode: str = "social_leads", 
    noise_level: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic price and social mentions time-series.
    Causality modes:
      - 'social_leads': Social mentions spike leads price jumps by 1-3 days.
      - 'price_leads': Price jump leads social mentions spike by 1-2 days.
      - 'no_causality': Price and social mentions are independent.
    """
    np.random.seed(seed)
    date_range = pd.date_range(end=datetime.datetime.now(), periods=days, freq="D")
    
    # 1. Base price process (Geometric Brownian Motion + shocks)
    price = 100.0
    prices = []
    mu = 0.0005  # slight upward drift
    sigma = 0.02  # daily volatility
    
    # 2. Social mentions baseline
    mentions_base = 500
    mentions = []
    
    # Shocks setup
    num_shocks = max(2, int(days / 20))
    shock_days = sorted(np.random.choice(range(10, days - 10), num_shocks, replace=False))
    
    # Pre-generate shocks impact
    price_shocks = np.zeros(days)
    social_shocks = np.zeros(days)
    
    for sd in shock_days:
        if causality_mode == "social_leads":
            # Social spikes first, price follows 1-3 days later
            social_shocks[sd] = np.random.uniform(3.0, 5.0)  # large multiplier
            delay = np.random.randint(1, 4)
            price_shocks[sd + delay] = np.random.uniform(0.08, 0.15)  # 8-15% jump
        elif causality_mode == "price_leads":
            # Price jumps first, social spikes 1-2 days later
            price_shocks[sd] = np.random.uniform(0.08, 0.15)
            delay = np.random.randint(1, 3)
            social_shocks[sd + delay] = np.random.uniform(3.0, 5.0)
        else:
            # Independent shocks
            price_shocks[sd] = np.random.uniform(0.08, 0.15)
            sd_social = np.random.randint(10, days - 10)
            social_shocks[sd_social] = np.random.uniform(3.0, 5.0)

    # Simulate path
    for i in range(days):
        # Calculate price
        price_ret = np.random.normal(mu, sigma)
        if price_shocks[i] > 0:
            price_ret += price_shocks[i]
        price *= np.exp(price_ret)
        prices.append(price)
        
        # Calculate social mentions
        mention_val = mentions_base * (1.0 + np.random.normal(0, noise_level))
        if social_shocks[i] > 0:
            mention_val *= social_shocks[i]
        # Keep positive
        mention_val = max(50, mention_val)
        mentions.append(mention_val)
        
    df = pd.DataFrame({
        "price": prices,
        "volume": np.random.uniform(1e6, 5e6, days) * (np.array(prices) / 100.0),
        "social_mentions": mentions
    }, index=date_range)
    
    # Add sentiment score (-1 to 1)
    # Sentiment correlates with price returns
    returns = df["price"].pct_change().fillna(0)
    sentiment = 0.1 + 2.0 * returns + np.random.normal(0, 0.15, days)
    df["sentiment"] = np.clip(sentiment, -1.0, 1.0)
    
    # Mark shock event days for event-study visualization
    df["is_event"] = False
    for sd in shock_days:
        df.iloc[sd, df.columns.get_loc("is_event")] = True
        
    return df

def get_combined_dataset(
    coin_id: str = "bitcoin", 
    days: int = 90, 
    causality_mode: str = "social_leads",
    noise_level: float = 0.2,
    use_real_price: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """
    Tries to fetch real price data from CoinGecko and aligns it with generated social data.
    If fetching fails or use_real_price is False, returns pure synthetic data.
    Returns (DataFrame, is_simulated_price).
    """
    df = None
    is_simulated = True
    
    if use_real_price:
        df = fetch_coingecko_data(coin_id, days)
        if df is not None:
            is_simulated = False
            # Generate aligned social mentions
            n = len(df)
            np.random.seed(42)
            
            # Simple simulation of social data matching the real price returns
            returns = df["price"].pct_change().fillna(0)
            
            # Base mentions
            social = np.random.normal(500, 100, n)
            
            # Inject relationship based on causality_mode
            if causality_mode == "social_leads":
                # Create a synthetic social signal that predicts price returns
                # We shift returns backwards to make social lead
                lead_days = 2
                shifted_returns = returns.shift(-lead_days).fillna(0)
                # social mentions spike before price returns spike
                social += np.maximum(0, shifted_returns) * 5000 + np.random.uniform(0, 200, n)
            elif causality_mode == "price_leads":
                # Price returns predict social spikes
                lag_days = 1
                lagged_returns = returns.shift(lag_days).fillna(0)
                social += np.maximum(0, lagged_returns) * 5000 + np.random.uniform(0, 200, n)
            else:
                # Independent social mentions
                # Just random spikes
                spikes = np.zeros(n)
                spike_idx = np.random.choice(range(n), max(1, int(n/20)), replace=False)
                spikes[spike_idx] = 1000
                social += spikes
                
            df["social_mentions"] = np.clip(social, 100, 10000)
            # Sentiment correlates with returns
            sentiment = 0.05 + 1.5 * returns + np.random.normal(0, 0.1, n)
            df["sentiment"] = np.clip(sentiment, -1.0, 1.0)
            
            # Mark event days based on local maxima of social mentions
            threshold = df["social_mentions"].quantile(0.95)
            df["is_event"] = df["social_mentions"] > threshold

    if df is None:
        df = generate_synthetic_data(days, causality_mode, noise_level)
        is_simulated = True
        
    return df, is_simulated
