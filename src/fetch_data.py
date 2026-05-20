"""
Fetch macroeconomic data from FRED for the Wall Street vs Main Street analysis.

Variables:
- S&P 500 (SP500) — stock market
- Michigan Consumer Sentiment (UMCSENT) — consumer confidence  
- CPI (CPIAUCSL) — inflation
- Federal Funds Rate (FEDFUNDS) — monetary policy
- Corporate Profits (CP) — who's winning
- Real Disposable Income (DSPIC96) — what consumers actually have
"""
import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Load your API key from .env
# os.getenv() reads the value from your .env file safely
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

print("Connected to FRED. Fetching data...")

# Each string in quotes is a FRED series ID
# You can look any of these up at fred.stlouisfed.org
series = {
   "nasdaq":             "NASDAQCOM",    # NASDAQ Composite Index, monthly, back to 1971
    "consumer_sentiment": "UMCSENT",     # U of Michigan consumer sentiment, monthly
    "cpi":                "CPIAUCSL",    # Consumer price index, monthly
    "fed_funds_rate":     "FEDFUNDS",    # Federal funds rate, monthly
    "corporate_profits":  "CP",          # Corporate profits, quarterly
    "real_disp_income":   "DSPIC96",     # Real disposable income, monthly
}

frames = {}
for name, code in series.items():
    print(f"  Fetching {name} ({code})...")
    frames[name] = fred.get_series(code, observation_start="2000-01-01")

print("\nDone! Here's what we got:")
for name, s in frames.items():
    print(f"  {name:<22}: {len(s):>5} observations, "
          f"{s.index[0].year}–{s.index[-1].year}, "
          f"frequency: {s.index.freq if s.index.freq else 'irregular'}")