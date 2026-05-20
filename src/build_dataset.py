"""
Build a clean monthly dataset for the Wall Street vs Main Street VAR analysis.

Key steps:
1. Resample everything to monthly frequency
2. Align all series to the same date range
3. Calculate derived variables (inflation rate, stock returns)
4. Check for and handle missing values
5. Save to data/clean_data.csv
"""
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# ── Fetch raw series ───────────────────────────────────────────────────────
print("Fetching data from FRED...")

nasdaq      = fred.get_series("NASDAQCOM",    observation_start="2000-01-01")
sentiment   = fred.get_series("UMCSENT",      observation_start="2000-01-01")
cpi         = fred.get_series("CPIAUCSL",     observation_start="2000-01-01")
fed_funds   = fred.get_series("FEDFUNDS",     observation_start="2000-01-01")
corp_profit = fred.get_series("CP",           observation_start="2000-01-01")
disp_income = fred.get_series("DSPIC96",      observation_start="2000-01-01")

# ── Resample to monthly ────────────────────────────────────────────────────
# ME = month end frequency
# NASDAQ is daily so we take the last trading day of each month
# Everything else is already monthly or quarterly — mean fills gaps
print("Resampling to monthly frequency...")

df = pd.DataFrame({
    "nasdaq":          nasdaq.resample("ME").last(),       # last price of month
    "sentiment":       sentiment.resample("ME").mean(),    # consumer confidence
    "cpi":             cpi.resample("ME").mean(),          # price level
    "fed_funds":       fed_funds.resample("ME").mean(),    # interest rate
    "corp_profit":     corp_profit.resample("ME").ffill(), # quarterly->monthly via forward fill
    "disp_income":     disp_income.resample("ME").mean(),  # real disposable income
})

# ── Derive key variables ───────────────────────────────────────────────────
# These are the variables economists actually model rather than raw levels

# Stock returns: % change in NASDAQ month over month
# Economists model returns rather than price levels — returns are stationary
df["stock_returns"] = df["nasdaq"].pct_change() * 100

# Inflation rate: year over year % change in CPI
# This is what people feel — not the price level but how fast it's rising
df["inflation_rate"] = df["cpi"].pct_change(12) * 100

# Sentiment change: month over month change in consumer confidence
# Captures the direction of mood shifts
df["sentiment_change"] = df["sentiment"].diff()

# Real disposable income growth: month over month % change
df["income_growth"] = df["disp_income"].pct_change() * 100

# ── Drop NaNs created by pct_change and diff ──────────────────────────────
# pct_change(12) creates 12 NaN rows at the start, so we lose the first year
df = df.dropna()

print(f"\nFinal dataset: {len(df)} months ({df.index[0].strftime('%b %Y')}–"
      f"{df.index[-1].strftime('%b %Y')})")
print(f"\nColumns: {list(df.columns)}")

# ── Sanity check ───────────────────────────────────────────────────────────
# These numbers should match what we know about economic history
print(f"\nSanity check:")
print(f"  Avg inflation 2000-2019 : "
      f"{df.loc[:'2019', 'inflation_rate'].mean():.1f}%")
print(f"  Avg inflation 2021-2024 : "
      f"{df.loc['2021':'2024', 'inflation_rate'].mean():.1f}%")
print(f"  Avg sentiment 2000-2019 : "
      f"{df.loc[:'2019', 'sentiment'].mean():.1f}")
print(f"  Avg sentiment 2021-2024 : "
      f"{df.loc['2021':'2024', 'sentiment'].mean():.1f}")
print(f"  NASDAQ avg monthly return: "
      f"{df['stock_returns'].mean():.2f}%")

# ── Save ───────────────────────────────────────────────────────────────────
df.to_csv("data/clean_data.csv")
print("\n✅ Saved to data/clean_data.csv")