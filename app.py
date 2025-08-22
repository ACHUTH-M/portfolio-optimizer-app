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
    eq = equity_curve(port_rets)
    avg = port_rets.mean()
    vol = port_rets.std(ddof=1)
    sharpe = (avg / vol) * np.sqrt(252) if vol and vol > 0 else np.nan
    cagr = (eq.iloc[-1] ** (252 / len(eq)) - 1.0) if len(eq) > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) > 0 else np.nan
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDD": mdd}, eq

# Robust loader that handles DD-MM-YYYY, YYYY-MM-DD, mixed, and unknown date column names
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df_raw = pd.read_csv(file)

    # Auto-detect date column (case-insensitive); fallback to the first col
    date_col = None
    for c in df_raw.columns:
        if any(k in c.lower() for k in ("date", "time", "timestamp")):
            date_col = c; break
    if date_col is None:
        date_col = df_raw.columns[0]

    # Parse dates (support mixed + dayfirst)
    df_raw[date_col] = pd.to_datetime(
        df_raw[date_col],
        errors="coerce",          # invalid -> NaT
        dayfirst=True,            # supports 13-01-2018 style
        format="mixed"            # pandas >= 2.0 mixed formats
    )
    df = df_raw.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Keep only numeric columns (silently drops strings like tickers)
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric asset columns found after parsing. "
                         "Ensure your CSV has numbers for asset columns.")

    # Decide: prices or returns?
    looks_like_prices = (num.max() > 5).any()
    if looks_like_prices:
        st.info("Detected prices — converting to simple daily returns.")
        num = num.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Make index continuous daily and ffill gaps (typical for markets)
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
    # simple daily rebalance with tc if you want:
    # r = r - tc * 0  # (no turnover here since w is constant)
    return r

def simulate_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    # start equal-weighted (align to columns!)
    w0 = pd.Series(np.ones(df.shape[1]) / df.shape[1], index=df.columns)

    # grow each asset with its own cumulative return
    eq = (1.0 + df).cumprod()

    # value weights each day (normalize across columns)
    wts = eq.div(eq.sum(axis=1), axis=0)

    # use previous day's weights to compute today's portfolio return
    wts_shifted = wts.shift()

    # for the very first day after shift (NaNs), use starting weights
    if len(wts_shifted) > 0:
        wts_shifted.iloc[0] = w0

    # portfolio daily return
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
    lookback = st.slider("Lookback (days) for Momentum / InvVol / MinVar", 20, 180, 60)

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

        # Metrics & equity curves
        metrics = []
        eq_curves = {}
        for name, r in strategies.items():
            # Align to common index (shorter series like MinVar start after lookback)
            aligned = r.loc[rets.index.intersection(r.index)]
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


