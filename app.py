# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# =========================
# Helpers & metrics
# =========================
def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()

def _ann_factor(freq_code: str):
    return 252 if freq_code == "D" else (52 if freq_code == "W" else 12)

def eval_portfolio(port_rets: pd.Series, weights_history=None, freq_code="D"):
    port_rets = port_rets.dropna()
    if port_rets.empty:
        nan = float("nan")
        return {"CAGR": nan, "Volatility": nan, "Sharpe": nan, "Sortino": nan,
                "Calmar": nan, "MaxDD": nan, "VaR5": nan, "CVaR5": nan, "Turnover": nan}, pd.Series(dtype=float)
    af = _ann_factor(freq_code)
    eq = equity_curve(port_rets)
    m = float(port_rets.mean())
    s = float(port_rets.std(ddof=1))
    neg = port_rets[port_rets < 0]
    d = float(neg.std(ddof=1)) if len(neg) > 1 else float("nan")
    sharpe = (m / s) * np.sqrt(af) if (s and s > 0) else np.nan
    sortino = (m / d) * np.sqrt(af) if (d and d > 0) else np.nan
    cagr = (eq.iloc[-1] ** (af / len(eq)) - 1.0)
    mdd_series = (eq / eq.cummax() - 1.0)
    maxdd = float(mdd_series.min())
    calmar = (cagr / abs(maxdd)) if maxdd < 0 else np.nan
    var5 = float(np.percentile(port_rets, 5))
    cvar5 = float(neg.mean()) if len(neg) else np.nan

    turnover = 0.0
    if isinstance(weights_history, list) and len(weights_history) > 1:
        diffs = [np.abs(weights_history[i] - weights_history[i-1]).sum() for i in range(1, len(weights_history))]
        turnover = float(np.mean(diffs))

    return {"CAGR": cagr, "Volatility": s, "Sharpe": sharpe, "Sortino": sortino,
            "Calmar": calmar, "MaxDD": maxdd, "VaR5": var5, "CVaR5": cvar5, "Turnover": turnover}, eq

def apply_rebalance_freq(rets: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    """Resample returns to the chosen rebalance frequency (by compounding)."""
    if freq_label == "Daily":
        return rets
    rule = {"Weekly": "W-FRI", "Monthly": "M"}[freq_label]
    eq = (1.0 + rets).cumprod()
    eq_f = eq.resample(rule).last().dropna(how="all")
    rets_f = eq_f.pct_change().dropna(how="any")
    return rets_f

# =========================
# Robust CSV loader
# =========================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df_raw = pd.read_csv(file)

    # Auto-detect date col (case-insensitive); fallback to first col
    date_col = None
    for c in df_raw.columns:
        if any(k in str(c).lower() for k in ("date", "time", "timestamp")):
            date_col = c; break
    if date_col is None:
        date_col = df_raw.columns[0]

    # Parse dates (mixed formats + day-first support)
    df_raw[date_col] = pd.to_datetime(
        df_raw[date_col],
        errors="coerce",
        dayfirst=True,
        format="mixed"  # pandas >= 2.0
    )
    df = (df_raw
          .dropna(subset=[date_col])
          .set_index(date_col)
          .sort_index())

    # Keep only numeric asset columns
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric asset columns found after parsing.")

    # Prices or returns?
    if (num.max() > 5).any():
        st.info("Detected prices — converting to simple daily returns.")
        num = num.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Continuous daily index & forward-fill
    all_days = pd.date_range(num.index.min(), num.index.max(), freq="D")
    num = num.reindex(all_days).ffill().dropna(how="all")
    num.index.name = "Date"
    num = num.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return num

# =========================
# Strategies (with TC support)
# =========================
def simulate_equal_weight(df: pd.DataFrame, tc=0.0):
    N = df.shape[1]
    w = np.ones(N)/N
    r = df @ w
    hist = [w.copy()] * len(df)  # constant weights → no turnover
    return r.astype(float), hist

def simulate_buy_and_hold(df: pd.DataFrame, tc=0.0):
    w0 = pd.Series(np.ones(df.shape[1]) / df.shape[1], index=df.columns)
    eq = (1.0 + df).cumprod()
    wts = eq.div(eq.sum(axis=1), axis=0)
    wts_shifted = wts.shift()
    if len(wts_shifted) > 0:
        wts_shifted.iloc[0] = w0
    hist = [w0.values]
    for i in range(1, len(wts_shifted)):
        hist.append(wts_shifted.iloc[i].values)
    r = (df * wts_shifted).sum(axis=1)
    w_arr = np.vstack(hist)
    turnover = np.abs(np.diff(w_arr, axis=0)).sum(axis=1)
    cost = np.insert(tc * turnover, 0, 0.0)
    r_net = r - cost
    return r_net.astype(float), [w for w in hist]

def simulate_naive_momentum(df: pd.DataFrame, lookback=60, tc=0.0):
    N = df.shape[1]
    w_curr = np.ones(N)/N
    hist = [w_curr.copy()]
    out = []
    roll = (1.0 + df).rolling(lookback).apply(lambda x: np.prod(x) - 1.0, raw=False)
    for t in range(len(df)):
        r_vec = df.iloc[t].values
        out.append(float(np.dot(w_curr, r_vec)))
        if t >= lookback-1:
            mom = roll.iloc[t].values
            pos = np.clip(mom, 0, None)
            w_new = pos/pos.sum() if pos.sum() > 0 else np.ones(N)/N
            out[-1] -= tc * np.abs(w_new - w_curr).sum()
            w_curr = w_new
        hist.append(w_curr.copy())
    return pd.Series(out, index=df.index), hist

def simulate_inverse_vol(df: pd.DataFrame, lookback=60, tc=0.0, eps=1e-8):
    N = df.shape[1]
    w_curr = np.ones(N)/N
    hist = [w_curr.copy()]
    out = []
    for t in range(len(df)):
        r_vec = df.iloc[t].values
        out.append(float(np.dot(w_curr, r_vec)))
        if t >= lookback-1:
            vol = df.iloc[t-lookback+1:t+1].std().values + eps
            inv = 1.0 / vol
            w_new = inv / inv.sum()
            out[-1] -= tc * np.abs(w_new - w_curr).sum()
            w_curr = w_new
        hist.append(w_curr.copy())
    return pd.Series(out, index=df.index), hist

def simulate_min_variance(df: pd.DataFrame, lookback=60, tc=0.0):
    N = df.shape[1]
    w_curr = np.ones(N)/N
    hist = [w_curr.copy()]
    out = []
    def minvar(returns: pd.DataFrame):
        cov = returns.cov().values
        n = cov.shape[0]
        cov = cov + np.eye(n) * 1e-8
        x0 = np.ones(n)/n
        cons = ({"type":"eq","fun": lambda x: np.sum(x)-1.0},)
        bounds = [(0,1)]*n
        res = minimize(lambda x: float(x @ cov @ x), x0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else x0
    for t in range(len(df)):
        r_vec = df.iloc[t].values
        out.append(float(np.dot(w_curr, r_vec)))
        if t >= lookback-1:
            w_new = minvar(df.iloc[t-lookback+1:t+1])
            out[-1] -= tc * np.abs(w_new - w_curr).sum()
            w_curr = w_new
        hist.append(w_curr.copy())
    return pd.Series(out, index=df.index), hist

def simulate_rl_placeholder(df: pd.DataFrame):
    np.random.seed(42)
    w = np.ones(df.shape[1]) / df.shape[1]
    noise = np.random.normal(0, 0.004, size=len(df))
    r = pd.Series(df.dot(w).values + noise, index=df.index)
    hist = [w.copy()] * len(df)
    return r, hist

# =========================
# UI — data & controls
# =========================
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
    st.write("**Tips:** First column must be a date; other columns numeric prices/returns.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    freq_label = st.selectbox("Rebalance frequency", ["Daily", "Weekly", "Monthly"], index=0)
    tc_bps = st.slider("Transaction cost (bps per 100% turnover)", 0, 50, 5, help="10 bps = 0.0010 cost")
    tc = tc_bps / 10000.0
    lookback = st.slider("Lookback for Momentum / Inverse Vol / MinVar", 20, 180, 60)
    enabled = st.multiselect(
        "Strategies to run",
        ["Equal Weight","Buy & Hold","Momentum","Inverse Vol","Min Variance","RL (placeholder)"],
        default=["Equal Weight","Buy & Hold","Momentum","Inverse Vol","Min Variance","RL (placeholder)"]
    )

# Apply chosen rebalance frequency
rets_use = apply_rebalance_freq(rets, freq_label)
freq_code = {"Daily":"D","Weekly":"W","Monthly":"M"}[freq_label]

# =========================
# Run backtests
# =========================
if st.button("Run backtests"):
    with st.spinner("Running strategies…"):
        sims = {}
        if "Equal Weight" in enabled:
            sims["Equal Weight"] = simulate_equal_weight(rets_use, tc=tc)
        if "Buy & Hold" in enabled:
            sims["Buy & Hold"] = simulate_buy_and_hold(rets_use, tc=tc)
        if "Momentum" in enabled:
            sims["Momentum"] = simulate_naive_momentum(rets_use, lookback=lookback, tc=tc)
        if "Inverse Vol" in enabled:
            sims["Inverse Vol"] = simulate_inverse_vol(rets_use, lookback=lookback, tc=tc)
        if "Min Variance" in enabled:
            sims["Min Variance"] = simulate_min_variance(rets_use, lookback=lookback, tc=tc)
        if "RL (placeholder)" in enabled:
            sims["RL (placeholder)"] = simulate_rl_placeholder(rets_use)

        # Common index for fair comparison
        common_idx = None
        for r, _w in sims.values():
            common_idx = r.index if common_idx is None else common_idx.intersection(r.index)

        # Metrics & curves
        rows, eq_curves = [], {}
        for name, (r, w_hist) in sims.items():
            aligned = r.reindex(common_idx).dropna()
            perf, eq = eval_portfolio(aligned, w_hist, freq_code=freq_code)
            rows.append([name, perf["CAGR"], perf["Volatility"], perf["Sharpe"],
                         perf["Sortino"], perf["Calmar"], perf["MaxDD"],
                         perf["VaR5"], perf["CVaR5"], perf["Turnover"]])
            eq_curves[name] = eq

        st.subheader("📊 Strategy Comparison")
        metrics_df = pd.DataFrame(rows, columns=[
            "Strategy","CAGR","Volatility","Sharpe","Sortino","Calmar","MaxDD","VaR5","CVaR5","Turnover"
        ])
        st.dataframe(metrics_df.style.format({
            "CAGR":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}","Sortino":"{:.2f}",
            "Calmar":"{:.2f}","MaxDD":"{:.2%}","VaR5":"{:.2%}","CVaR5":"{:.2%}","Turnover":"{:.2f}"
        }))

        st.download_button("⬇️ Download metrics (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="metrics.csv", mime="text/csv")

        st.subheader("📈 Equity Curves")
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)

        st.download_button("⬇️ Download equity curves (CSV)",
                           # -------------------------
# Allocation pie chart
# -------------------------
st.subheader("🍰 Average Allocation by Strategy")

# Let the user pick a strategy to inspect
strategy_for_pie = st.selectbox("Select strategy", list(sims.keys()))

# Grab its returns and weight history
r_sel, w_hist_sel = sims[strategy_for_pie]

# Build a weight dataframe aligned to the strategy's own index,
# then restrict to the common comparison window.
# Many strategies stored len(df)+1 weights, so trim to len(r).
W = np.vstack(w_hist_sel[:len(r_sel)])
wdf = pd.DataFrame(W, index=r_sel.index[:len(W)], columns=rets_use.columns)
wdf = wdf.reindex(common_idx).dropna(how="any")

# Average allocation over time (and also show the latest if you like)
avg_w = wdf.mean(axis=0)
latest_w = wdf.iloc[-1]

col1, col2 = st.columns(2)

with col1:
    st.caption("Average weights over the evaluation window")
    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    ax1.pie(avg_w.values, labels=avg_w.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

with col2:
    st.caption("Last rebalanced weights")
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    ax2.pie(latest_w.values, labels=latest_w.index, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

# Optional CSV downloads for documentation
pie_df = pd.DataFrame({"AverageWeight": avg_w, "LastWeight": latest_w})
st.download_button("⬇️ Download weights (CSV)",
                   data=pie_df.to_csv().encode("utf-8"),
                   file_name=f"{strategy_for_pie.replace(' ','_').lower()}_weights.csv",
                   mime="text/csv")

            data=eq_df.to_csv().encode("utf-8"),
            file_name="equity_curves.csv", mime="text/csv")


