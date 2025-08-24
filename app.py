# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

    # Find date column
    date_col = None
    for c in df_raw.columns:
        if any(k in str(c).lower() for k in ("date", "time", "timestamp")):
            date_col = c
            break
    if date_col is None:
        date_col = df_raw.columns[0]

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True, format="mixed")
    df = (df_raw.dropna(subset=[date_col]).set_index(date_col).sort_index())

    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric asset columns found after parsing.")

    if (num.max() > 5).any():
        st.info("Detected prices — converting to simple daily returns.")
        num = num.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    all_days = pd.date_range(num.index.min(), num.index.max(), freq="D")
    num = num.reindex(all_days).ffill().dropna(how="all")
    num.index.name = "Date"
    num = num.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return num

# =========================
# Strategies
# =========================
def simulate_equal_weight(df: pd.DataFrame, tc=0.0):
    N = df.shape[1]
    w = np.ones(N)/N
    r = df @ w
    hist = [w.copy()] * len(df)
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
    return r.astype(float), [w for w in hist]

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
# UI
# =========================
st.title("Portfolio Optimizer with Baselines & RL")

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
    st.stop()

with st.sidebar:
    st.header("Settings")
    freq_label = st.selectbox("Rebalance frequency", ["Daily", "Weekly", "Monthly"], index=0)
    lookback = st.slider("Lookback (days) for Momentum / Inverse Vol / MinVar", 20, 180, 60)
    enabled = st.multiselect(
        "Strategies to run",
        ["Equal Weight","Buy & Hold","Momentum","Inverse Vol","Min Variance","RL (placeholder)"],
        default=["Equal Weight","Buy & Hold","Momentum","Inverse Vol","Min Variance","RL (placeholder)"]
    )

rets_use = apply_rebalance_freq(rets, freq_label)
freq_code = {"Daily":"D","Weekly":"W","Monthly":"M"}[freq_label]

if st.button("Run backtests"):
    with st.spinner("Running strategies…"):
        sims = {}
        if "Equal Weight" in enabled:
            sims["Equal Weight"] = simulate_equal_weight(rets_use)
        if "Buy & Hold" in enabled:
            sims["Buy & Hold"] = simulate_buy_and_hold(rets_use)
        if "Momentum" in enabled:
            sims["Momentum"] = simulate_naive_momentum(rets_use, lookback=lookback)
        if "Inverse Vol" in enabled:
            sims["Inverse Vol"] = simulate_inverse_vol(rets_use, lookback=lookback)
        if "Min Variance" in enabled:
            sims["Min Variance"] = simulate_min_variance(rets_use, lookback=lookback)
        if "RL (placeholder)" in enabled:
            sims["RL (placeholder)"] = simulate_rl_placeholder(rets_use)

        common_idx = None
        for r, _w in sims.values():
            common_idx = r.index if common_idx is None else common_idx.intersection(r.index)

        metrics_rows, eq_curves = [], {}
        for name, (r, w_hist) in sims.items():
            aligned = r.reindex(common_idx).dropna()
            perf, eq = eval_portfolio(aligned, w_hist, freq_code=freq_code)
            metrics_rows.append([name, perf["CAGR"], perf["Volatility"], perf["Sharpe"],
                                 perf["Sortino"], perf["Calmar"], perf["MaxDD"],
                                 perf["VaR5"], perf["CVaR5"], perf["Turnover"]])
            eq_curves[name] = eq

        st.subheader("Strategy Comparison")
        metrics_df = pd.DataFrame(metrics_rows, columns=[
            "Strategy","CAGR","Volatility","Sharpe","Sortino","Calmar","MaxDD","VaR5","CVaR5","Turnover"
        ])
        st.dataframe(metrics_df.style.format({
            "CAGR":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}","Sortino":"{:.2f}",
            "Calmar":"{:.2f}","MaxDD":"{:.2%}","VaR5":"{:.2%}","CVaR5":"{:.2%}","Turnover":"{:.2f}"
        }))

        st.subheader("Equity Curves")
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)

        # ======================
        # Pie chart section
        # ======================
        st.subheader("Average Allocation by Strategy")
        strategy_for_pie = st.selectbox("Select strategy for pie chart", list(sims.keys()))
        r_sel, w_hist_sel = sims[strategy_for_pie]
        W = np.vstack(w_hist_sel[:len(r_sel)])
        wdf = pd.DataFrame(W, index=r_sel.index[:len(W)], columns=rets_use.columns)
        wdf = wdf.reindex(common_idx).dropna(how="any")

        avg_w = wdf.mean(axis=0)
        latest_w = wdf.iloc[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Average weights over the evaluation window")
            fig1, ax1 = plt.subplots(figsize=(4.5,4.5))
            ax1.pie(avg_w.values, labels=avg_w.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)
        with col2:
            st.caption("Last rebalanced weights")
            fig2, ax2 = plt.subplots(figsize=(4.5,4.5))
            ax2.pie(latest_w.values, labels=latest_w.index, autopct="%1.1f%%", startangle=90)
            ax2.axis("equal")
            st.pyplot(fig2)
            # ======================================================
# Live Portfolio (beta)
# ======================================================
st.markdown("---")
st.header("Live Portfolio (beta)")

with st.expander("Live settings"):
    st.write("Enter tickers (comma-separated) and choose a refresh rate.")
    tickers_text = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA, GOOGL")
    live_mode = st.checkbox("Enable live auto-refresh", value=False,
                            help="Refresh the chart automatically while this page is open.")
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=10, max_value=300, value=60, step=10)
    price_interval = st.selectbox("Price interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d"], index=6,
                                  help="Note: very short intervals might be limited/delayed on some tickers/exchanges.")
    lookback_days = st.slider("Lookback (days) for the live plot", 1, 365, 180, help="How far back to show the live chart")
    equal_w = st.checkbox("Use equal weights", value=True)
    custom_w_text = st.text_input("Custom weights (optional, comma-separated, sums to 1)", value="")

# Optional auto-refresh (does nothing unless live_mode=True)
if live_mode:
    st.autorefresh(interval=refresh_sec * 1000, key="live_portfolio_refresher")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_prices(tickers: list[str], lookback: int, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV (Adj Close) with yfinance; return price DataFrame (Adj Close).
    Cached for `ttl` seconds to lighten API calls.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    # yfinance period must be a string like '30d'. We’ll request more than lookback to be safe.
    period = f"{max(lookback, 5)}d"
    data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, progress=False)
    # yfinance returns multi-index columns for multiple tickers; normalize to a simple wide DF
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy() if "Close" in data else data["Adj Close"].copy()
    else:
        prices = data.copy()
    prices = prices.dropna(how="all")
    # Some single-ticker returns as Series; make it DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.columns = [c.split(" ")[0] for c in prices.columns]  # tidy names
    # Restrict to last N days
    if lookback > 0 and len(prices) > 0:
        cutoff = prices.index.max() - pd.Timedelta(days=lookback)
        prices = prices[prices.index >= cutoff]
    return prices

def parse_weights(tickers: list[str], equal: bool, custom_text: str) -> pd.Series:
    if equal or not custom_text.strip():
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers, dtype=float)
        return w
    # parse custom weights
    try:
        vals = [float(x) for x in custom_text.split(",")]
        vals = np.array(vals, dtype=float)
        if len(vals) != len(tickers):
            raise ValueError("Number of weights must match number of tickers.")
        if np.isclose(vals.sum(), 0.0):
            raise ValueError("Weights sum to zero.")
        vals = vals / vals.sum()
        return pd.Series(vals, index=tickers, dtype=float)
    except Exception as e:
        st.warning(f"Could not parse custom weights: {e}. Falling back to equal weights.")
        return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers, dtype=float)

# Build the live portfolio
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
if len(tickers) == 0:
    st.info("Enter at least one ticker above to see a live portfolio.")
else:
    prices = fetch_prices(tickers, lookback=lookback_days, interval=price_interval)
    if prices.empty:
        st.warning("No live price data returned. Try a longer interval (e.g., 1d) or different tickers.")
    else:
        # Compute live returns
        rets_live = prices.pct_change().dropna(how="any")
        weights_live = parse_weights(tickers, equal_w, custom_w_text).reindex(rets_live.columns).fillna(0.0)

        # Portfolio returns + equity
        port_live = (rets_live @ weights_live).astype(float)
        eq_live = (1.0 + port_live).cumprod()
        latest_ts = eq_live.index.max()
        # Show numbers
        colA, colB = st.columns([3, 2])

        with colA:
            st.subheader("Live Portfolio Equity")
            st.line_chart(eq_live, height=320)
            st.caption(f"Last update: **{latest_ts}** (timezone = data source)")

        with colB:
            st.subheader("Current Allocation")
            # Current weights based on latest prices (value-weighted with the chosen static target)
            # For this simple live view, we plot the target weights; you could switch to value weights if desired.
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.pie(weights_live.values, labels=weights_live.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # Daily stats block
        if len(port_live) > 0:
            daily_ret = float(port_live.iloc[-1])
            st.metric("Latest Period Return", f"{daily_ret:.2%}")
        # Optional: show last few price rows for debugging
        with st.expander("Show raw latest prices"):
            st.dataframe(prices.tail())




