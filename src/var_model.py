"""
Vector Autoregression (VAR) Model
Wall Street vs Main Street: Why is the stock market soaring
while consumer confidence is low and inflation is rising?

Steps:
1. Test for stationarity (ADF test) — VAR requires stationary variables
2. Select optimal lag length — how many months back to look
3. Fit the VAR model
4. Run Granger causality tests — does X statistically predict Y?
5. Generate Impulse Response Functions — if inflation spikes, what happens?
6. Run Chow structural break test — did relationships change after 2020?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/clean_data.csv", index_col=0, parse_dates=True)
print(f"Loaded {len(df)} months of data "
      f"({df.index[0].strftime('%b %Y')}–{df.index[-1].strftime('%b %Y')})")

# ── Select variables for VAR ───────────────────────────────────────────────
# We use the derived stationary variables, not raw levels
# stock_returns, inflation_rate, sentiment, fed_funds, income_growth
var_vars = ["stock_returns", "inflation_rate", "sentiment", 
            "fed_funds", "income_growth"]
data = df[var_vars].copy()

# ── Step 1: Stationarity Tests (ADF) ──────────────────────────────────────
# VAR models require stationary series — no trends or unit roots
# ADF null hypothesis: series has a unit root (non-stationary)
# p < 0.05 means we reject the null — series is stationary
print("\n" + "=" * 55)
print("STEP 1: STATIONARITY TESTS (Augmented Dickey-Fuller)")
print("=" * 55)
print("p < 0.05 = stationary ✅  |  p > 0.05 = non-stationary ⚠️\n")

for col in var_vars:
    result = adfuller(data[col].dropna())
    status = "✅" if result[1] < 0.05 else "⚠️ "
    print(f"  {col:<20}: p={result[1]:.3f}  {status}")

# ── Step 2: Lag Selection ──────────────────────────────────────────────────
# The lag order tells VAR how many months back to look
# AIC (Akaike Info Criterion) balances fit vs complexity
# Lower AIC = better model
print("\n" + "=" * 55)
print("STEP 2: LAG LENGTH SELECTION")
print("=" * 55)

model  = VAR(data.dropna())
result = model.select_order(maxlags=12)
print(result.summary())

# Use AIC optimal lag
aic_lag = result.aic
print(f"\nOptimal lag by AIC: {aic_lag}")

# Cap at 6 to avoid overfitting with our sample size
lag = min(aic_lag, 6)
print(f"Using lag: {lag}")

# ── Step 3: Fit VAR model ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: VAR MODEL RESULTS")
print("=" * 55)

var_model   = VAR(data.dropna())
var_results = var_model.fit(lag)
print(var_results.summary())

# ── Step 4: Granger Causality Tests ───────────────────────────────────────
# Granger causality: does knowing X help predict Y better than Y alone?
# This is NOT true causality — it's predictive causality
# p < 0.05 means X Granger-causes Y
print("\n" + "=" * 55)
print("STEP 4: GRANGER CAUSALITY TESTS")
print("=" * 55)
print("Does X help predict Y? (p < 0.05 = yes)\n")

pairs = [
    ("inflation_rate", "sentiment",    "Does inflation predict sentiment?"),
    ("inflation_rate", "stock_returns","Does inflation predict stock returns?"),
    ("fed_funds",      "stock_returns","Does Fed rate predict stock returns?"),
    ("fed_funds",      "sentiment",    "Does Fed rate predict sentiment?"),
    ("sentiment",      "stock_returns","Does sentiment predict stock returns?"),
    ("stock_returns",  "sentiment",    "Do stock returns predict sentiment?"),
]

for cause, effect, question in pairs:
    test_data = data[[effect, cause]].dropna()
    result    = grangercausalitytests(test_data, maxlag=lag, verbose=False)
    # Use F-test p-value at our chosen lag
    p_val  = result[lag][0]["ssr_ftest"][1]
    status = "✅ Yes" if p_val < 0.05 else "❌ No"
    print(f"  {question}")
    print(f"    p={p_val:.3f}  →  {status}\n")

# ── Step 5: Impulse Response Functions ────────────────────────────────────
# IRF: if we shock one variable by 1 std dev, how do others respond over time?
# This is the most powerful output of a VAR model
print("\n" + "=" * 55)
print("STEP 5: IMPULSE RESPONSE FUNCTIONS")
print("=" * 55)
print("Generating IRFs — 24 month horizon...")

irf = var_results.irf(24)

# Plot IRFs for the two key shocks
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#1C2833")

DARK  = "#1C2833"
PANEL = "#2C3E50"
BLUE  = "#2C5F8A"
ORANGE= "#E07B39"
RED   = "#C0392B"
GREEN = "#3A8A5C"
GRAY  = "#95A5A6"
WHITE = "#FFFFFF"

def style_irf(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A4A4A")
    ax.grid(axis="y", color="#4A4A4A", linewidth=0.5, linestyle="--")
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")

# Get variable indices
idx = {v: i for i, v in enumerate(var_vars)}

# IRF data
irf_data = irf.irfs  # shape: (periods, n_vars, n_vars)

periods = range(25)

# Plot 1: Shock to inflation → effect on sentiment
ax = axes[0, 0]
style_irf(ax, "Inflation Shock → Consumer Sentiment")
ax.plot(periods, irf_data[:, idx["sentiment"],    idx["inflation_rate"]], 
        color=ORANGE, linewidth=2.5)
ax.fill_between(periods,
    irf.cum_effect_stderr()[:, idx["sentiment"], idx["inflation_rate"]] * -1.96,
    irf.cum_effect_stderr()[:, idx["sentiment"], idx["inflation_rate"]] *  1.96,
    alpha=0.2, color=ORANGE)
ax.set_xlabel("Months after shock")
ax.set_ylabel("Response")

# Plot 2: Shock to inflation → effect on stock returns
ax = axes[0, 1]
style_irf(ax, "Inflation Shock → Stock Returns")
ax.plot(periods, irf_data[:, idx["stock_returns"], idx["inflation_rate"]], 
        color=RED, linewidth=2.5)
ax.set_xlabel("Months after shock")
ax.set_ylabel("Response")

# Plot 3: Shock to Fed funds → effect on sentiment
ax = axes[1, 0]
style_irf(ax, "Fed Rate Hike → Consumer Sentiment")
ax.plot(periods, irf_data[:, idx["sentiment"],    idx["fed_funds"]], 
        color=BLUE, linewidth=2.5)
ax.set_xlabel("Months after shock")
ax.set_ylabel("Response")

# Plot 4: Shock to sentiment → effect on stock returns
ax = axes[1, 1]
style_irf(ax, "Sentiment Shock → Stock Returns")
ax.plot(periods, irf_data[:, idx["stock_returns"], idx["sentiment"]], 
        color=GREEN, linewidth=2.5)
ax.set_xlabel("Months after shock")
ax.set_ylabel("Response")

fig.suptitle("Impulse Response Functions\nHow Shocks Ripple Through the Economy",
             color=WHITE, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/impulse_response.png", dpi=150, 
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("✅ Saved outputs/impulse_response.png")

# ── Step 6: Structural Break Test (Chow Test) ─────────────────────────────
# Did the relationship between these variables change after COVID?
# We split the sample at 2020-01 and compare regression coefficients
print("\n" + "=" * 55)
print("STEP 6: STRUCTURAL BREAK ANALYSIS (Pre vs Post 2020)")
print("=" * 55)

pre  = data[data.index <  "2020-01-01"]
post = data[data.index >= "2020-01-01"]

print(f"Pre-2020  sample: {len(pre)} months")
print(f"Post-2020 sample: {len(post)} months")

# Compare correlations between key variables pre and post 2020
print("\nCorrelation between stock returns and sentiment:")
pre_corr  = pre["stock_returns"].corr(pre["sentiment"])
post_corr = post["stock_returns"].corr(post["sentiment"])
print(f"  Pre-2020  : {pre_corr:.3f}")
print(f"  Post-2020 : {post_corr:.3f}")
change = "WEAKENED" if abs(post_corr) < abs(pre_corr) else "STRENGTHENED"
print(f"  → Relationship has {change} since 2020")

print("\nCorrelation between inflation and sentiment:")
pre_corr2  = pre["inflation_rate"].corr(pre["sentiment"])
post_corr2 = post["inflation_rate"].corr(post["sentiment"])
print(f"  Pre-2020  : {pre_corr2:.3f}")
print(f"  Post-2020 : {post_corr2:.3f}")

print("\n✅ VAR analysis complete!")