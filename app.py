import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    # If looks like prices → convert to returns
    if (df.max(numeric_only=True) > 5).any():
        df = df.pct_change().dropna()
    return df

st.title("📈 Portfolio Optimizer with Baselines & RL")

uploaded = st.file_uploader("Upload your returns.csv or prices.csv", type=["csv"])
if uploaded:
    rets = load_data(uploaded)
    st.success(f"Loaded {rets.shape[0]} rows and {rets.shape[1]} assets")
    st.dataframe(rets.head())
else:
    st.info("Upload a CSV file to continue.")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def equity_curve(r):
    return (1 + r).cumprod()

def eval_portfolio(port_rets):
    eq = equity_curve(port_rets)
    avg = port_rets.mean()
    vol = port_rets.std()
    sharpe = avg/vol * np.sqrt(252) if vol > 0 else np.nan
    cagr = eq.iloc[-1] ** (252/len(port_rets)) - 1
    mdd = (eq/eq.cummax() - 1).min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDD": mdd}, eq

# -------------------------
# Baseline strategies
# -------------------------
def simulate_equal_weight(df):
    w = np.ones(df.shape[1]) / df.shape[1]
    rets = df.dot(w)
    return rets

def simulate_buy_and_hold(df):
    w = np.ones(df.shape[1]) / df.shape[1]
    eq = (1 + df).cumprod()
    wts = eq.div(eq.sum(axis=1), axis=0)
    rets = (df * wts.shift().fillna(w)).sum(axis=1)
    return rets

def simulate_naive_momentum(df, lookback=60):
    roll = (1 + df).rolling(lookback).apply(lambda x: np.prod(x)-1, raw=False)
    mom = (roll > 0).astype(int)
    mom = mom.div(mom.sum(axis=1).replace(0, np.nan), axis=0).fillna(1/df.shape[1])
    rets = (df * mom.shift().fillna(1/df.shape[1])).sum(axis=1)
    return rets

def simulate_inverse_vol(df, lookback=60):
    vol = df.rolling(lookback).std()
    inv = 1/vol
    wts = inv.div(inv.sum(axis=1), axis=0)
    rets = (df * wts.shift().fillna(1/df.shape[1])).sum(axis=1)
    return rets

def simulate_min_variance(df, lookback=60):
    def minvar(returns):
        cov = returns.cov().values
        n = cov.shape[0]
        x0 = np.ones(n)/n
        cons = {"type":"eq", "fun": lambda x: np.sum(x)-1}
        bounds = [(0,1)]*n
        res = minimize(lambda x: x @ cov @ x, x0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else x0

    weights = []
    rets_out = []
    for t in range(lookback, len(df)):
        w = minvar(df.iloc[t-lookback:t])
        weights.append(w)
        rets_out.append(np.dot(w, df.iloc[t].values))
    return pd.Series(rets_out, index=df.index[lookback:])

# -------------------------
# Placeholder PPO model
# -------------------------
def simulate_rl(df):
    np.random.seed(42)
    w = np.ones(df.shape[1]) / df.shape[1]
    noise = np.random.normal(0, 0.01, size=len(df))
    rets = df.dot(w) + noise
    return pd.Series(rets, index=df.index)

# -------------------------
# Run strategies
# -------------------------
strategies = {
    "Equal Weight": simulate_equal_weight(rets),
    "Buy & Hold": simulate_buy_and_hold(rets),
    "Momentum": simulate_naive_momentum(rets),
    "Inverse Vol": simulate_inverse_vol(rets),
    "Min Variance": simulate_min_variance(rets),
    "RL (PPO Placeholder)": simulate_rl(rets),
}

# -------------------------
# Show results
# -------------------------
st.header("📊 Strategy Comparison")
fig, ax = plt.subplots(figsize=(10,5))

metrics = []
for name, r in strategies.items():
    perf, eq = eval_portfolio(r)
    metrics.append([name, perf["CAGR"], perf["Volatility"], perf["Sharpe"], perf["MaxDD"]])
    eq.plot(ax=ax, label=name)

ax.set_title("Equity Curves")
ax.legend()
st.pyplot(fig)

metrics_df = pd.DataFrame(metrics, columns=["Strategy","CAGR","Volatility","Sharpe","MaxDD"])
st.dataframe(metrics_df)
