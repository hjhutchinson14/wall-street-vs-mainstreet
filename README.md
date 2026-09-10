[wall-street-vs-mainstreet-README.md](https://github.com/user-attachments/files/32064842/wall-street-vs-mainstreet-README.md)
# Wall Street vs. Main Street: Decoupling in the U.S. Economy

**Henry Hutchinson | UMass Amherst MSBA 2027**

A VAR (Vector Autoregression) econometric model examining the growing disconnect between U.S. stock market performance and consumer sentiment. Uses Granger causality testing, impulse response analysis, and structural break detection on monthly FRED data to ask: when equity markets rally, does Main Street feel it?

---

## Research Question

> *Does U.S. stock market performance predict improvements in consumer sentiment and income growth — or has the relationship broken down, and if so, when?*

This project tests the "wealth effect" hypothesis empirically: the idea that rising asset prices translate into consumer confidence and spending. Using VAR methods, I examine the direction and magnitude of predictive relationships across five macroeconomic variables before and after the COVID-19 structural break.

---

## Data

Monthly U.S. macroeconomic data from **FRED (Federal Reserve Bank of St. Louis)**.

| Variable | Description |
|---|---|
| Stock Returns | S&P 500 monthly return (%) |
| Inflation | CPI YoY % change |
| Consumer Sentiment | University of Michigan Consumer Sentiment Index |
| Fed Funds Rate | Federal funds effective rate (%) |
| Income Growth | Real disposable personal income (YoY %) |

---

## Methodology

### 6-Step VAR Pipeline

**1. Stationarity Testing**
Augmented Dickey-Fuller (ADF) tests on all series. First-differencing applied where required to ensure valid VAR estimation.

**2. Lag Selection**
Optimal lag length selected via Akaike Information Criterion (AIC), capped at 6 lags to preserve degrees of freedom given monthly sample size.

**3. VAR Estimation**
Reduced-form VAR fitted on the full sample. All variables modeled jointly to capture feedback dynamics that univariate models miss.

**4. Granger Causality Testing**
Six variable pairs tested for predictive causality:

| Pair | Hypothesis Tested |
|---|---|
| Stock Returns → Consumer Sentiment | Does the market predict confidence? |
| Inflation → Consumer Sentiment | Does rising prices erode sentiment? |
| Fed Rate → Consumer Sentiment | Do rate hikes dampen consumer outlook? |
| Income Growth → Consumer Sentiment | Does income drive confidence? |
| Stock Returns → Income Growth | Does market wealth reach workers? |
| Inflation → Income Growth | Does inflation erode real income? |

*Note: Granger causality is predictive, not structural — it identifies whether lagged values of X improve forecasts of Y.*

**5. Impulse Response Functions (IRFs)**
24-month horizon IRFs trace how a one-standard-deviation shock to each variable propagates through the system. Key IRFs: Inflation Shock → Consumer Sentiment, Fed Rate Hike → Consumer Sentiment, Stock Return Shock → Income Growth.

**6. Structural Break Analysis**
Pre-2020 vs. post-2020 correlation structure compared to test whether the pandemic fundamentally altered the relationship between equity markets and consumer outcomes.

---

## Key Results

- Granger causality from stock returns to consumer sentiment is statistically significant in the pre-2020 period, consistent with the wealth effect
- Post-2020 correlations show a structural shift: equity rallies decoupled from sentiment improvements, likely reflecting K-shaped recovery dynamics
- Inflation shocks predict sentiment declines with a 2–3 month lag
- Fed rate hikes transmit to consumer sentiment over a 6–9 month horizon

---

## Skills Demonstrated

- **VAR modeling**: Lag selection, reduced-form estimation, joint dynamics
- **Granger causality**: F-test framework, multi-pair testing
- **Impulse response analysis**: 24-month shock propagation
- **Structural break detection**: Pre/post-2020 correlation comparison
- **Python**: `statsmodels`, `pandas`, `matplotlib`
- **FRED API**: Monthly time-series retrieval and stationarity transformation

---

## Project Structure

```
wall-street-vs-mainstreet/
├── src/
│   └── var_model.py        # Full VAR pipeline — stationarity through IRFs
├── outputs/                # IRF plots, causality test tables, correlation matrices
└── requirements.txt
```

---

## Background

The 2020–2024 period produced one of the most extreme divergences between equity market performance and household economic wellbeing in modern U.S. history. This project started as a question I kept running into in my economics coursework: the aggregate data suggested a strong economy while the micro data suggested most households weren't feeling it. VAR is the right tool for this question because it doesn't impose a causal direction — it lets the data tell you which variables predict which.

---

## A Note on Tooling

Code scaffolding and boilerplate were built with AI assistance (Claude by Anthropic). The research design, variable selection, model specification, and all analytical conclusions are my own.

---

*Data: FRED (Federal Reserve Bank of St. Louis) | Model: VAR + Granger Causality | Author: Henry Hutchinson*
