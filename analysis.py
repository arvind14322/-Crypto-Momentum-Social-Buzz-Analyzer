import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
import statsmodels.api as sm
from typing import Dict, Tuple, Any, List

def compute_cross_correlation(
    df: pd.DataFrame, 
    col1: str = "social_mentions", 
    col2: str = "price", 
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Computes cross-correlation between col1 and col2 returns/changes.
    Returns a DataFrame with lag and correlation coefficient.
    Positive lag means col1 leads col2.
    """
    # Use returns/percentage change to ensure stationarity
    s1 = df[col1].pct_change().dropna() if col1 == "price" else df[col1].diff().dropna()
    s2 = df[col2].pct_change().dropna() if col2 == "price" else df[col2].diff().dropna()
    
    # Align indices
    common_idx = s1.index.intersection(s2.index)
    s1 = s1.loc[common_idx]
    s2 = s2.loc[common_idx]
    
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    
    for lag in lags:
        if lag < 0:
            # col1 lags col2 (col1 shifted back, correlates with past col2)
            c = s1.corr(s2.shift(-lag))
        elif lag > 0:
            # col1 leads col2 (col2 shifted back, correlates with past col1)
            c = s1.shift(lag).corr(s2)
        else:
            c = s1.corr(s2)
        corrs.append(c)
        
    return pd.DataFrame({"lag": lags, "correlation": corrs})

def run_granger_causality(
    df: pd.DataFrame, 
    social_col: str = "social_mentions", 
    price_col: str = "price", 
    max_lag: int = 5
) -> Dict[str, Any]:
    """
    Performs Granger causality tests.
    Tests:
      1. Does Social Granger-cause Price Returns?
      2. Does Price Returns Granger-cause Social?
    Returns dictionary with p-values for each lag.
    """
    results = {
        "social_causes_price": {},
        "price_causes_social": {},
        "stationarity_check": {}
    }
    
    # Prepare stationary series
    price_ret = df[price_col].pct_change().dropna()
    
    # Diff social mentions if not stationary
    # Simple check: diff mentions
    social_diff = df[social_col].diff().dropna()
    
    # Align series
    test_df = pd.DataFrame({
        "price_ret": price_ret,
        "social_diff": social_diff
    }).dropna()
    
    if len(test_df) < (max_lag * 3):
        return {"error": "Insufficient data points for causality test"}
        
    # 1. Social causes Price: [dependent=price_ret, independent=social_diff]
    try:
        g_test1 = grangercausalitytests(test_df[["price_ret", "social_diff"]], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            # Extract p-value of SSR-based F-test
            p_val = g_test1[lag][0]["ssr_ftest"][1]
            results["social_causes_price"][lag] = p_val
    except Exception as e:
        results["social_causes_price"]["error"] = str(e)
        
    # 2. Price causes Social: [dependent=social_diff, independent=price_ret]
    try:
        g_test2 = grangercausalitytests(test_df[["social_diff", "price_ret"]], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            p_val = g_test2[lag][0]["ssr_ftest"][1]
            results["price_causes_social"][lag] = p_val
    except Exception as e:
        results["price_causes_social"]["error"] = str(e)
        
    return results

def fit_garch_model(df: pd.DataFrame, price_col: str = "price") -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Fits GARCH(1,1) model to price returns.
    Returns:
      - Series of conditional volatility.
      - Dictionary with model parameters/status.
    """
    returns = df[price_col].pct_change().dropna() * 100  # Scale returns for optimization stability
    
    try:
        model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        fit_res = model.fit(disp="off")
        
        # Unscale the volatility back
        vol = fit_res.conditional_volatility / 100.0
        
        summary_info = {
            "converged": fit_res.fit_start is not None,
            "aic": fit_res.aic,
            "bic": fit_res.bic,
            "params": fit_res.params.to_dict()
        }
        return vol, summary_info
    except Exception as e:
        # Fallback to simple rolling historical volatility if GARCH fails
        rolling_vol = returns.rolling(window=7).std().fillna(method="bfill") / 100.0
        return rolling_vol, {"converged": False, "error": str(e)}

def run_event_study(
    df: pd.DataFrame, 
    price_col: str = "price", 
    event_col: str = "is_event", 
    window_pre: int = 3, 
    window_post: int = 5
) -> pd.DataFrame:
    """
    Analyzes price returns around social spike events (event_col == True).
    Aligns events on day 0 and returns DataFrame with Mean and Cumulative returns.
    """
    returns = df[price_col].pct_change().fillna(0)
    event_indices = np.where(df[event_col])[0]
    
    event_runs = []
    
    for idx in event_indices:
        # Check window bounds
        if idx - window_pre >= 0 and idx + window_post < len(df):
            # Extract return slice
            ret_slice = returns.iloc[idx - window_pre : idx + window_post + 1].values
            event_runs.append(ret_slice)
            
    if not event_runs:
        # Return empty shell
        days = list(range(-window_pre, window_post + 1))
        return pd.DataFrame({"day": days, "mean_return": 0.0, "cum_return": 0.0})
        
    event_matrix = np.array(event_runs)
    mean_returns = np.mean(event_matrix, axis=0)
    
    # Calculate cumulative returns starting from day -window_pre
    cum_returns = np.cumsum(mean_returns)
    # Alternatively, normalize cumulative abnormal returns so that day -1 or day 0 starts at 0
    # Let's do simple cumulative sum of mean returns
    
    days = list(range(-window_pre, window_post + 1))
    return pd.DataFrame({
        "day": days,
        "mean_return": mean_returns,
        "cum_return": cum_returns
    })
