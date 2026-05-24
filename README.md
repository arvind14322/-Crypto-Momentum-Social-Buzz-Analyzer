# Crypto Momentum & Social Buzz Analyzer

An interactive analytical tool to analyze relationships between cryptocurrency prices and social media metrics using statistical models.

## Features

- **Data Fetcher**: Connects to CoinGecko public API with simulated social buzz generation.
- **Cross-Correlation**: Analyzes lead-lag relationship of metrics at various daily lags.
- **Granger Causality**: Tests whether social buzz changes predict price returns (or vice versa).
- **GARCH Modeling**: Estimates asset price conditional volatility.
- **Event Study**: Measures price reactions around peak buzz days (t=0).

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run local validation checks:
   ```bash
   python verify_analysis.py
   ```

## Running the App

Start the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
