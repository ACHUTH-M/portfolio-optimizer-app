# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# -------------------------
# Utilities
# -------------------------
def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()

def eval_portfolio(port_rets: pd.Series):
    port_rets = port_rets.dropna()
    if port_rets.empty:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}, pd.Series(dtype=float)

    eq = equity_curve(port_rets)
    avg = float(port_rets.mean())
    vol = float(port_rets.std(ddof=1))
    sharpe = (avg / vol) * np.sqrt(252) if (np.isfinite(vol) and vol > 0) else np.nan
    cagr = (eq.iloc[-1] ** (252 / len(eq)) - 1.0) if len(eq) > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) > 0 else np.nan
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDD": mdd}, eq

# Robust loader that handles DD‑MM‑YYYY, YYYY‑MM‑DD, mixed, and unknown date column names
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df_raw = pd.read_csv(file)

    # Auto-detect date column (case-insensitive); fallback to the first column
    date_col = None
    for c in df_raw.columns:
        if any(k in str(c).lower() for k in ("date", "time", "timestamp")):
            date_col = c
            break
    if date_col is None:
        date_col = df_raw.columns[0]

    # Parse dates (support mixed + dayfirst for 13-01-2018 style)
    df_raw[date_col] = pd.to_datetime(
        df_raw[date_col],
        errors="coerce",
        dayfirst=True,
        format="mixed"  # pandas >= 2.0
    )
    df = (
        df_raw
        .dropna(subset=[date_col])
        .set_index(date_col)
        .sort_index()
    )

    # Keep only numeric columns (drop any non-numeric like tickers)
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric asset columns found after parsing. "
                         "Ensure your CSV has numeric prices/returns after the date column.")

    # Decide: prices or returns?
    looks_like_prices = (num.max() > 5).any()
    if looks_like_prices:
        st.info("Detected prices — converting to simple daily returns.")
        num = num.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Fill calendar gaps with ffill (markets often have missing days)
    all_days = pd.date_range(num.index.min(), num.index.max(), freq="D")
    num = num.reindex(all_days).ffill().dropna(how="all")
    num.index.name = "Date"

    # Final clean
    num = num.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return num

# -------------------------
# Baselines
# -------------------------
def simulate_equal_weight(df: pd.DataFrame, tc=0.0005) -> pd.Series:
    w = np.ones(df.shape[1]) / df.shape[1]
    r = (df @ w).astype(float)
    # No transaction costs modeled here (constant weights)
    return r

def simulate_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    # Start equal-weighted (align to columns to avoid fill errors)
    w0 = pd.Series(np.ones(df.shape[1]) / df.shape[1], index=df.columns)

    # Grow each asset with its own cumulative return
    eq = (1.0 + df).cumprod()

    # Value weights each day (normalize across columns)
    wts = eq.div(eq.sum(axis=1), axis=0)

    # Use previous day's weights to compute today's return
    wts_shifted = wts.shift()

    # For the first day (NaNs after shift), use starting weights
    if len(wts_shifted) > 0:
        # Assign a Series to the first row (safe, column-aligned)
        wts_shifted.iloc[0] = w0

    r = (df * wts_shifted).sum(axis=1)
    return r

def simulate_naive_momentum(df: pd.DataFrame, lookback=60) -> pd.Series:
    roll = (1.0 + df).rolling(lookback).apply(lambda x: np.prod(x) - 1.0, raw=False)
    pos = (roll > 0).astype(int)
    wts = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / df.shape[1])
    r = (df * wts.shift().fillna(1.0 / df.shape[1])).sum(axis=1)
    return r

def simulate_inverse_vol(df: pd.DataFrame, lookback=60) -> pd.Series:
    vol = df.rolling(lookback).std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    wts = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / df.shape[1])
    r = (df * wts.shift().fillna(1.0 / df.shape[1])).sum(axis=1)
    return r

def simulate_min_variance(df: pd.DataFrame, lookback=60) -> pd.Series:
    def minvar(returns: pd.DataFrame):
        cov = returns.cov().values
        n = cov.shape[0]
        cov = cov + np.eye(n) * 1e-8  # ridge for stability
        x0 = np.ones(n) / n
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
        bounds = [(0, 1)] * n
        res = minimize(lambda x: float(x @ cov @ x), x0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else x0

    out = []
    idx = df.index[lookback:]
    for t in range(lookback, len(df)):
        w = minvar(df.iloc[t - lookback:t])
        out.append(float(np.dot(w, df.iloc[t].values)))
    return pd.Series(out, index=idx)

def simulate_rl_placeholder(df: pd.DataFrame) -> pd.Series:
    np.random.seed(42)
    w = np.ones(df.shape[1]) / df.shape[1]
    noise = np.random.normal(0, 0.004, size=len(df))
    return pd.Series(df.dot(w).values + noise, index=df.index)

# -------------------------
# UI
# -------------------------
st.title("📈 Portfolio Optimizer with Baselines & RL")

uploaded = st.file_uploader("Upload your returns.csv or prices.csv", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to continue.")
    st.stop()

try:
    rets = load_data(uploaded)
    st.success(f"Loaded {rets.shape[0]} rows × {rets.shape[1]} assets.")
    st.dataframe(rets.head())
except Exception as e:
    st.error(f"Failed to read your CSV. Details: {e}")
    st.write("**Tips:** Ensure the first column is a date (e.g., 'Date') and other columns are numeric prices/returns.")
    st.stop()

with st.expander("Options"):
    lookback = st.slider("Lookback (days) for Momentum / Inverse Vol / MinVar", 20, 180, 60)

if st.button("Run backtests"):
    with st.spinner("Running strategies…"):
        strategies = {
            "Equal Weight": simulate_equal_weight(rets),
            "Buy & Hold": simulate_buy_and_hold(rets),
            "Momentum": simulate_naive_momentum(rets, lookback),
            "Inverse Vol": simulate_inverse_vol(rets, lookback),
            "Min Variance": simulate_min_variance(rets, lookback),
            "RL (placeholder)": simulate_rl_placeholder(rets),
        }

        # Metrics & equity curves (align to a common index)
        common_idx = None
        for r in strategies.values():
            common_idx = r.index if common_idx is None else common_idx.intersection(r.index)

        metrics = []
        eq_curves = {}
        for name, r in strategies.items():
            aligned = r.reindex(common_idx).dropna()
            perf, eq = eval_portfolio(aligned)
            metrics.append([name, perf["CAGR"], perf["Volatility"], perf["Sharpe"], perf["MaxDD"]])
            eq_curves[name] = eq

        st.subheader("📊 Strategy Comparison")
        metrics_df = pd.DataFrame(metrics, columns=["Strategy", "CAGR", "Volatility", "Sharpe", "MaxDD"])
        st.dataframe(metrics_df.style.format({
            "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}"
        }))

        st.subheader("📈 Equity Curves")
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)

