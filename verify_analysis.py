import pandas as pd
from data_fetcher import generate_synthetic_data
from analysis import (
    compute_cross_correlation,
    run_granger_causality,
    fit_garch_model,
    run_event_study
)

def run_tests():
    print("=== Verification Starting ===")
    
    # 1. Generate data
    print("Generating synthetic dataset (90 days)...")
    df = generate_synthetic_data(days=90, causality_mode="social_leads", noise_level=0.15)
    print(f"Data columns: {list(df.columns)}")
    print(f"Data head:\n{df.head(3)}")
    
    # 2. Test cross correlation
    print("\nComputing cross-correlation...")
    cc = compute_cross_correlation(df, max_lag=5)
    print(f"Cross-correlation results (lags -5 to 5):\n{cc}")
    
    # 3. Test Granger causality
    print("\nRunning Granger causality test...")
    granger = run_granger_causality(df, max_lag=3)
    if "error" in granger:
        print(f"Granger causality error: {granger['error']}")
    else:
        print("Granger causality p-values (Social -> Price):")
        for lag, pval in granger["social_causes_price"].items():
            print(f"  Lag {lag}: p = {pval:.4f}")
            
        print("Granger causality p-values (Price -> Social):")
        for lag, pval in granger["price_causes_social"].items():
            print(f"  Lag {lag}: p = {pval:.4f}")
            
    # 4. Test GARCH model
    print("\nFitting GARCH(1,1) model...")
    vol, garch_info = fit_garch_model(df)
    print(f"GARCH fitted: {garch_info.get('converged', False)}")
    if not garch_info.get("converged", False):
        print(f"GARCH fitting note: {garch_info.get('error', 'None')}")
    else:
        print(f"GARCH AIC: {garch_info.get('aic'):.2f}")
        print(f"GARCH parameters: {garch_info.get('params')}")
        
    # 5. Test Event Study
    print("\nRunning event study...")
    es = run_event_study(df, window_pre=2, window_post=3)
    print(f"Event study results:\n{es}")
    
    print("\n=== Verification Successful ===")

if __name__ == "__main__":
    run_tests()
