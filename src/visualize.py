"""
Wall Street vs Main Street — Dashboard
Visualizing the disconnect between markets, inflation, and consumer confidence
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/clean_data.csv", index_col=0, parse_dates=True)

var_vars    = ["stock_returns", "inflation_rate", "sentiment",
               "fed_funds", "income_growth"]
data        = df[var_vars].dropna()
var_model   = VAR(data)
var_results = var_model.fit(2)
irf         = var_results.irf(24)
irf_data    = irf.irfs
idx         = {v: i for i, v in enumerate(var_vars)}

# ── Colors & style ─────────────────────────────────────────────────────────
DARK  = "#1C2833"
PANEL = "#2C3E50"
BLUE  = "#2C5F8A"
ORANGE= "#E07B39"
RED   = "#C0392B"
GREEN = "#3A8A5C"
GRAY  = "#95A5A6"
WHITE = "#FFFFFF"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A4A4A")
    ax.grid(axis="y", color="#4A4A4A", linewidth=0.5, linestyle="--")

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

fig.suptitle(
    "Wall Street vs Main Street\n"
    "Why is the Market Soaring While Consumers Struggle?",
    fontsize=16, fontweight="bold", color=WHITE, y=0.98
)

# ── Panel 1: NASDAQ vs Consumer Sentiment over time ───────────────────────
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Panel 1: Stock Market vs Consumer Sentiment (2001–2025)")

ax1_twin = ax1.twinx()
ax1_twin.set_facecolor(PANEL)

# Normalize both to 2001 = 100 for comparison
nasdaq_norm    = df["nasdaq"]   / df["nasdaq"].iloc[0]   * 100
sentiment_norm = df["sentiment"]/ df["sentiment"].iloc[0]* 100

ax1.plot(df.index, nasdaq_norm,    color=ORANGE, linewidth=2,   label="NASDAQ (left)")
ax1_twin.plot(df.index, sentiment_norm, color=BLUE, linewidth=2, 
              linestyle="--", label="Consumer Sentiment (right)")

# Shade post-2020
ax1.axvspan(pd.Timestamp("2020-01-01"), df.index[-1],
            alpha=0.1, color=RED, label="Post-2020")
ax1.axvline(pd.Timestamp("2020-01-01"), color=RED, 
            linewidth=1, linestyle=":")
ax1.text(pd.Timestamp("2020-03-01"), nasdaq_norm.max() * 0.95,
         "COVID →", color=RED, fontsize=8)

ax1.set_ylabel("NASDAQ Index (2001=100)", color=ORANGE)
ax1_twin.set_ylabel("Sentiment Index (2001=100)", color=BLUE)
ax1_twin.tick_params(colors=WHITE, labelsize=8)
ax1_twin.yaxis.label.set_color(BLUE)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           facecolor=PANEL, labelcolor=WHITE, fontsize=8, loc="upper left")

# ── Panel 2: Inflation rate over time ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "Panel 2: Inflation Rate")

ax2.fill_between(df.index, df["inflation_rate"], 
                 where=df["inflation_rate"] > 2,
                 color=RED, alpha=0.4, label="Above 2% target")
ax2.fill_between(df.index, df["inflation_rate"],
                 where=df["inflation_rate"] <= 2,
                 color=GREEN, alpha=0.4, label="At/below target")
ax2.plot(df.index, df["inflation_rate"], color=WHITE, linewidth=1.5)
ax2.axhline(2, color=GRAY, linewidth=1, linestyle="--", label="Fed 2% target")
ax2.set_ylabel("YoY CPI %", color=WHITE)
ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7)

# ── Panel 3: Granger causality summary chart ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Panel 3: Granger Causality\n(p-values, lower = stronger)")

pairs = [
    ("Inflation\n→ Sentiment",      0.049),
    ("Inflation\n→ Stocks",         0.010),
    ("Fed Rate\n→ Stocks",          0.832),
    ("Fed Rate\n→ Sentiment",       0.456),
    ("Sentiment\n→ Stocks",         0.412),
    ("Stocks\n→ Sentiment",         0.000),
]
labels   = [p[0] for p in pairs]
p_values = [p[1] for p in pairs]
colors   = [GREEN if p < 0.05 else RED for p in p_values]

bars = ax3.barh(labels, p_values, color=colors, alpha=0.85)
ax3.axvline(0.05, color=WHITE, linewidth=1.5, 
            linestyle="--", label="p=0.05 threshold")
ax3.set_xlabel("p-value")
ax3.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)
ax3.tick_params(axis="y", labelsize=7)

# ── Panel 4: IRF — Inflation shock → Sentiment ────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Panel 4: IRF — Inflation Shock\n→ Consumer Sentiment")

periods  = range(25)
response = irf_data[:, idx["sentiment"], idx["inflation_rate"]]
ax4.plot(periods, response, color=ORANGE, linewidth=2.5)
ax4.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
ax4.fill_between(periods, response, 0,
                 where=[r < 0 for r in response],
                 alpha=0.3, color=RED, label="Negative effect")
ax4.fill_between(periods, response, 0,
                 where=[r >= 0 for r in response],
                 alpha=0.3, color=GREEN, label="Positive effect")
ax4.set_xlabel("Months after shock")
ax4.set_ylabel("Response")
ax4.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

# ── Panel 5: IRF — Stocks → Sentiment ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "Panel 5: IRF — Stock Market Shock\n→ Consumer Sentiment")

response2 = irf_data[:, idx["sentiment"], idx["stock_returns"]]
ax5.plot(periods, response2, color=BLUE, linewidth=2.5)
ax5.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
ax5.fill_between(periods, response2, 0,
                 where=[r >= 0 for r in response2],
                 alpha=0.3, color=GREEN, label="Positive effect")
ax5.fill_between(periods, response2, 0,
                 where=[r < 0 for r in response2],
                 alpha=0.3, color=RED, label="Negative effect")
ax5.set_xlabel("Months after shock")
ax5.set_ylabel("Response")
ax5.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

# ── Panel 6: Pre vs Post 2020 correlation heatmaps ────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
style_ax(ax6, "Panel 6: How Correlations Changed Pre vs Post 2020")

pre  = data[data.index <  "2020-01-01"][["stock_returns","inflation_rate",
                                          "sentiment","fed_funds"]]
post = data[data.index >= "2020-01-01"][["stock_returns","inflation_rate",
                                          "sentiment","fed_funds"]]

pre_corr  = pre.corr()
post_corr = post.corr()
diff_corr = post_corr - pre_corr

short_names = ["Stocks", "Inflation", "Sentiment", "Fed Rate"]
pre_corr.columns  = short_names
pre_corr.index    = short_names
post_corr.columns = short_names
post_corr.index   = short_names
diff_corr.columns = short_names
diff_corr.index   = short_names

import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn

im = ax6.imshow(diff_corr.values, cmap="RdYlGn", vmin=-0.5, vmax=0.5,
                aspect="auto")
ax6.set_xticks(range(4))
ax6.set_yticks(range(4))
ax6.set_xticklabels(short_names, color=WHITE, fontsize=9)
ax6.set_yticklabels(short_names, color=WHITE, fontsize=9)

for i in range(4):
    for j in range(4):
        val = diff_corr.values[i, j]
        ax6.text(j, i, f"{val:+.2f}", ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")

plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).ax.tick_params(colors=WHITE)
ax6.set_title("Panel 6: Change in Correlations Post-2020\n"
              "(green = stronger positive, red = stronger negative)",
              color=WHITE, fontsize=10, fontweight="bold")

# ── Panel 7: Fed funds rate vs inflation ──────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, "Panel 7: Fed Rate vs Inflation\nThe Policy Response")

ax7.plot(df.index, df["inflation_rate"], color=RED,  
         linewidth=2, label="Inflation Rate")
ax7.plot(df.index, df["fed_funds"],      color=BLUE, 
         linewidth=2, label="Fed Funds Rate", linestyle="--")
ax7.axvline(pd.Timestamp("2020-01-01"), color=GRAY, 
            linewidth=1, linestyle=":")
ax7.set_ylabel("Percent (%)", color=WHITE)
ax7.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

# ── Footer ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "Data: Federal Reserve FRED — NASDAQ, U of Michigan Consumer Sentiment, "
         "BLS CPI, Federal Funds Rate, Real Disposable Income  |  "
         "Model: VAR(2) with Granger Causality & Impulse Response Functions",
         ha="center", fontsize=7, color=GRAY)

plt.savefig("outputs/wall_street_vs_mainstreet.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("✅ Saved outputs/wall_street_vs_mainstreet.png")