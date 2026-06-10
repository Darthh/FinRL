# -*- coding: utf-8 -*-
"""
mu_multi_strategy_backtest.py
Runs backtests for 20 trading strategies on MU and MUU (2x Leverage)
for a 1-year period (June 2025 to June 2026), trains PPO/SAC models,
and saves the comparison results.
"""
from __future__ import annotations
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# Force CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {device}")
if device == "cuda":
    print(f"         GPU Device Name: {torch.cuda.get_device_name(0)}")

# Helper functions
def compute_rsi(price, period=14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ── Config ──────────────────────────────────────────────────────────────────────
TICKER = "MU"
TRAIN_START = "2015-01-01"  # Train on 10 years for robust RL agents
TRAIN_END   = "2025-12-01"
TEST_START  = "2025-12-01"
TEST_END    = "2026-06-01"   # Exactly 6-month backtest

INITIAL_CAPITAL = 100_000
TURBULENCE_THRESHOLD = 150

# Technical indicator list
INDICATORS_TO_USE = INDICATORS  # macd, boller bands, rsi_30, cci_30, dx_30, etc.

# ── Step 1: Download data ─────────────────────────────────────────────────────
print(f"\n[1/6] Downloading {TICKER} data...")
# Download a wider window to calculate rolling indicators without NaNs at the start of backtest
download_start = (datetime.strptime(TRAIN_START, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
df_raw = yf.download(TICKER, start=download_start, end=TEST_END)

# Convert multi-index columns if present (new yfinance behavior)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

df_raw = df_raw.reset_index()
# Rename columns to match YahooDownloader output structure
df_raw = df_raw.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjcp",
    "Volume": "volume"
})
df_raw["tic"] = TICKER
df_raw["day"] = pd.to_datetime(df_raw["date"]).dt.dayofweek
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")

# ── Step 2: Calculate Technical Indicators ─────────────────────────────────────
print("\n[2/6] Calculating technical indicators...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS_TO_USE,
    use_turbulence=True,
    user_defined_feature=False,
)
df_processed = fe.preprocess_data(df_raw)
df_processed = df_processed.sort_values(["date"]).reset_index(drop=True)

# Additional indicators needed for custom rules
close = df_processed["close"]
df_processed["rsi_14"] = compute_rsi(close, 14)
df_processed["sma_50"] = close.rolling(50).mean()
df_processed["sma_200"] = close.rolling(200).mean()

# Calculate realized volatilities (30-day and 90-day annualized)
daily_returns = close.pct_change()
df_processed["rv_30d"] = daily_returns.rolling(30).std() * np.sqrt(252) * 100
df_processed["rv_90d"] = daily_returns.rolling(90).std() * np.sqrt(252) * 100

# Earnings dates for MU: June 24, 2025; Sept 24, 2025; Dec 18, 2025; March 26, 2026; June 24, 2026.
earnings_dates = [
    datetime(2025, 6, 24),
    datetime(2025, 9, 24),
    datetime(2025, 12, 18),
    datetime(2026, 3, 26),
    datetime(2026, 6, 24)
]

def check_earnings_proximity(date_str, window_days=10):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    for ed in earnings_dates:
        # Check if dt is within window_days before earnings date, or up to 2 days after
        delta = (ed - dt).days
        if -2 <= delta <= window_days:
            return 1
    return 0

df_processed["earnings_proximity"] = df_processed["date"].apply(lambda d: check_earnings_proximity(d, 10))

# ── Step 3: Simulate $MUU (2x Leverage) ───────────────────────────────────────
print("\n[3/6] Simulating MUU (2x Daily Leveraged MU)...")
# Daily returns of MU
df_processed["mu_daily_ret"] = df_processed["close"].pct_change().fillna(0)
# 2x Daily return for MUU
df_processed["muu_daily_ret"] = 2 * df_processed["mu_daily_ret"]

# Reconstruct MUU price series starting from MU price at the beginning of the dataset
muu_close = []
curr_price = df_processed["close"].iloc[0]
for ret in df_processed["muu_daily_ret"]:
    curr_price = curr_price * (1 + ret)
    muu_close.append(curr_price)
df_processed["close_MUU"] = muu_close

# Split train/test
train_df = data_split(df_processed, TRAIN_START, TRAIN_END)
test_df  = data_split(df_processed, TEST_START,  TEST_END)
print(f"      Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# ── Step 4: Train RL Agents (PPO and SAC) on MU ───────────────────────────────
print("\n[4/6] Training RL Agents (PPO & SAC) on MU...")
stock_dim   = 1
state_space = 1 + 2 * stock_dim + len(INDICATORS_TO_USE) * stock_dim

env_kwargs_full = {
    "hmax": 1000,
    "initial_amount": INITIAL_CAPITAL,
    "buy_cost_pct":  [0.001],
    "sell_cost_pct": [0.001],
    "reward_scaling": 1e-4,
    "num_stock_shares": [0],
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS_TO_USE,
    "action_space": stock_dim,
}

# Env for training
e_train = StockTradingEnv(df=train_df, **env_kwargs_full)
env_train, _ = e_train.get_sb_env()
agent = DRLAgent(env=env_train)

# Train PPO (Fast)
print("      Training PPO agent...")
model_ppo = agent.get_model("ppo", model_kwargs={"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64})
trained_ppo = agent.train_model(model=model_ppo, tb_log_name="mu_ppo_multi", total_timesteps=20000)
trained_ppo.save(os.path.join(TRAINED_MODEL_DIR, "mu_ppo_multi"))

# Train SAC (Fast)
print("      Training SAC agent...")
model_sac = agent.get_model("sac", model_kwargs={"batch_size": 128, "buffer_size": 100_000, "learning_starts": 500})
trained_sac = agent.train_model(model=model_sac, tb_log_name="mu_sac_multi", total_timesteps=20000)
trained_sac.save(os.path.join(TRAINED_MODEL_DIR, "mu_sac_multi"))

# ── Step 5: Simulate all 20 Strategies on Test Data ───────────────────────────
print("\n[5/6] Simulating 20 strategies...")
test_dates = test_df["date"].tolist()
n_days = len(test_df)

# Prepare lists to hold portfolio values
portfolios = {f"S{i}": [INITIAL_CAPITAL] * n_days for i in range(1, 21)}

# Pull vectors for easy computation
close_MU = test_df["close"].values
close_MUU = test_df["close_MUU"].values
rsi = test_df["rsi_14"].values
sma50 = test_df["sma_50"].values
sma200 = test_df["sma_200"].values
macd = test_df["macd"].values
macd_signal = test_df["close"].rolling(9).mean().values # approximation or use standard from fe
# Replace with actual signal column if available
if "macd" in test_df.columns:
    # standard MACD = 12-26 EMA. We can calculate signal line
    macd_series = test_df["macd"]
    macd_sig = macd_series.rolling(9).mean().fillna(0).values
else:
    macd_series = np.zeros(n_days)
    macd_sig = np.zeros(n_days)
macd = macd_series.values if hasattr(macd_series, "values") else macd_series

boll_ub = test_df["boll_ub"].values if "boll_ub" in test_df.columns else (close_MU + 2 * close_MU.std())
boll_lb = test_df["boll_lb"].values if "boll_lb" in test_df.columns else (close_MU - 2 * close_MU.std())
rv30 = test_df["rv_30d"].values
rv90 = test_df["rv_90d"].values
earn_prox = test_df["earnings_proximity"].values

# Fetch RL predictions
# Run PPO
e_test = StockTradingEnv(df=test_df, turbulence_threshold=TURBULENCE_THRESHOLD, **env_kwargs_full)
df_account_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_test)
ppo_values = df_account_ppo["account_value"].values[:n_days]

# Run SAC
e_test = StockTradingEnv(df=test_df, turbulence_threshold=TURBULENCE_THRESHOLD, **env_kwargs_full)
df_account_sac, _ = DRLAgent.DRL_prediction(model=trained_sac, environment=e_test)
sac_values = df_account_sac["account_value"].values[:n_days]

# ── Run Rule-Based Backtests ──
for t in range(1, n_days):
    mu_ret = (close_MU[t] - close_MU[t-1]) / close_MU[t-1]
    muu_ret = (close_MUU[t] - close_MUU[t-1]) / close_MUU[t-1]
    
    # ── S1: B&H MU ──
    portfolios["S1"][t] = portfolios["S1"][t-1] * (1 + mu_ret)
    
    # ── S2: B&H MUU ──
    portfolios["S2"][t] = portfolios["S2"][t-1] * (1 + muu_ret)
    
    # ── S3: RSI Buy/Sell MU (Buy <30, Sell >70) ──
    # State tracking: S3_holding (shares or cash)
    # We can simulate this using a simple asset allocation fraction: 1.0 (long) or 0.0 (cash)
    s3_alloc = 0.0
    for prev_t in range(t):
        if rsi[prev_t] < 30:
            s3_alloc = 1.0
        elif rsi[prev_t] > 70:
            s3_alloc = 0.0
    portfolios["S3"][t] = portfolios["S3"][t-1] * (1 + s3_alloc * mu_ret)

    # ── S4: RSI Buy/Sell MUU ──
    s4_alloc = 0.0
    for prev_t in range(t):
        if rsi[prev_t] < 30:
            s4_alloc = 1.0
        elif rsi[prev_t] > 70:
            s4_alloc = 0.0
    portfolios["S4"][t] = portfolios["S4"][t-1] * (1 + s4_alloc * muu_ret)

    # ── S5: RSI Short Mean Reversion MU (Short >70, Cover <30) ──
    s5_alloc = 0.0  # -1.0 (short) or 0.0 (cash)
    for prev_t in range(t):
        if rsi[prev_t] > 70:
            s5_alloc = -1.0
        elif rsi[prev_t] < 30:
            s5_alloc = 0.0
    portfolios["S5"][t] = portfolios["S5"][t-1] * (1 + s5_alloc * mu_ret)

    # ── S6: RSI Short Mean Reversion MUU ──
    s6_alloc = 0.0
    for prev_t in range(t):
        if rsi[prev_t] > 70:
            s6_alloc = -1.0
        elif rsi[prev_t] < 30:
            s6_alloc = 0.0
    portfolios["S6"][t] = portfolios["S6"][t-1] * (1 + s6_alloc * muu_ret)

    # ── S7: SMA 50/200 Cross MU ──
    s7_alloc = 1.0 if sma50[t-1] > sma200[t-1] else 0.0
    portfolios["S7"][t] = portfolios["S7"][t-1] * (1 + s7_alloc * mu_ret)

    # ── S8: SMA 50/200 Cross MUU ──
    s8_alloc = 1.0 if sma50[t-1] > sma200[t-1] else 0.0
    portfolios["S8"][t] = portfolios["S8"][t-1] * (1 + s8_alloc * muu_ret)

    # ── S9: MACD Cross MU ──
    s9_alloc = 1.0 if macd[t-1] > macd_sig[t-1] else 0.0
    portfolios["S9"][t] = portfolios["S9"][t-1] * (1 + s9_alloc * mu_ret)

    # ── S10: MACD Cross MUU ──
    s10_alloc = 1.0 if macd[t-1] > macd_sig[t-1] else 0.0
    portfolios["S10"][t] = portfolios["S10"][t-1] * (1 + s10_alloc * muu_ret)

    # ── S11: BB Mean Reversion MU (Buy <lower, Sell >upper) ──
    s11_alloc = 0.0
    for prev_t in range(t):
        if close_MU[prev_t] < boll_lb[prev_t]:
            s11_alloc = 1.0
        elif close_MU[prev_t] > boll_ub[prev_t]:
            s11_alloc = 0.0
    portfolios["S11"][t] = portfolios["S11"][t-1] * (1 + s11_alloc * mu_ret)

    # ── S12: BB Mean Reversion MUU ──
    s12_alloc = 0.0
    for prev_t in range(t):
        if close_MU[prev_t] < boll_lb[prev_t]:
            s12_alloc = 1.0
        elif close_MU[prev_t] > boll_ub[prev_t]:
            s12_alloc = 0.0
    portfolios["S12"][t] = portfolios["S12"][t-1] * (1 + s12_alloc * muu_ret)

    # ── S13: Volatility Breakout MU (Buy 30d > 90d RV) ──
    s13_alloc = 1.0 if rv30[t-1] > rv90[t-1] else 0.0
    portfolios["S13"][t] = portfolios["S13"][t-1] * (1 + s13_alloc * mu_ret)

    # ── S14: Volatility Breakout MUU ──
    s14_alloc = 1.0 if rv30[t-1] > rv90[t-1] else 0.0
    portfolios["S14"][t] = portfolios["S14"][t-1] * (1 + s14_alloc * muu_ret)

    # ── S15: Earnings Play MU (Buy 10d before, sell 2d after) ──
    s15_alloc = 1.0 if earn_prox[t] else 0.0
    portfolios["S15"][t] = portfolios["S15"][t-1] * (1 + s15_alloc * mu_ret)

    # ── S16: Earnings Play MUU ──
    s16_alloc = 1.0 if earn_prox[t] else 0.0
    portfolios["S16"][t] = portfolios["S16"][t-1] * (1 + s16_alloc * muu_ret)

    # ── S17: Cash-Secured Put / Covered Call Wheel MU ──
    # Every 5 days (weekly), we simulate options premium capture
    # If price fell > 2% week-on-week, we get assigned. If it rose > 2%, we get called away.
    # Collect 1.0% premium weekly, otherwise gain/lose based on share status.
    # We can model this weekly options logic:
    # State tracking: S17_holding_shares (True/False)
    # Simple mathematical approximation of option premiums + assignment:
    # Portfolio value increases by weekly premium (1% / 5 days = ~0.2% daily)
    # Plus if holding shares, gains/losses of MU.
    # Let's write the step-by-step state machine:
    pass

# Run state machine options wheel backtest for S17 and S18
s17_shares = 0
s17_cash = INITIAL_CAPITAL
s18_shares = 0
s18_cash = INITIAL_CAPITAL
weekly_premium_pct = 0.01  # 1% weekly premium for MU
weekly_premium_muu_pct = 0.02  # 2% weekly premium for MUU (higher vol)

for t in range(n_days):
    price = close_MU[t]
    price_muu = close_MUU[t]
    
    # Weekly check (every 5 trading days)
    if t > 0 and t % 5 == 0:
        prev_price = close_MU[t-5]
        prev_price_muu = close_MUU[t-5]
        
        # S17 MU Wheel
        if s17_shares == 0:
            # Sell Put (collect premium)
            premium = s17_cash * weekly_premium_pct
            s17_cash += premium
            # Check assignment
            if price < prev_price * 0.98:  # Assigned if price drops >2%
                s17_shares = s17_cash / price
                s17_cash = 0
        else:
            # Sell Covered Call (collect premium)
            premium = (s17_shares * price) * (weekly_premium_pct * 0.8) # call premium slightly lower
            s17_cash += premium
            # Check call-away
            if price > prev_price * 1.02:  # Called away if price rises >2%
                s17_cash += s17_shares * price
                s17_shares = 0
                
        # S18 MUU Wheel
        if s18_shares == 0:
            premium = s18_cash * weekly_premium_muu_pct
            s18_cash += premium
            if price_muu < prev_price_muu * 0.96: # Assigned if price drops >4% (leveraged)
                s18_shares = s18_cash / price_muu
                s18_cash = 0
        else:
            premium = (s18_shares * price_muu) * (weekly_premium_muu_pct * 0.8)
            s18_cash += premium
            if price_muu > prev_price_muu * 1.04: # Called away if price rises >4%
                s18_cash += s18_shares * price_muu
                s18_shares = 0

    # Daily portfolio value updates
    portfolios["S17"][t] = s17_cash + s17_shares * price
    portfolios["S18"][t] = s18_cash + s18_shares * price_muu

# ── RL Predictions Integration ──
for t in range(n_days):
    # S19: PPO Agent MU
    portfolios["S19"][t] = ppo_values[t] if t < len(ppo_values) else portfolios["S19"][t-1]
    # S20: SAC Agent MU
    portfolios["S20"][t] = sac_values[t] if t < len(sac_values) else portfolios["S20"][t-1]


# ── Step 6: Compute performance metrics ────────────────────────────────────────
print("\n[6/6] Computing performance metrics...")
strategy_metadata = {
    "S1":  {"name": "S1: Buy & Hold MU (Benchmark)", "type": "Benchmark", "underlying": "MU"},
    "S2":  {"name": "S2: Buy & Hold MUU (Benchmark)", "type": "Leveraged", "underlying": "MUU"},
    "S3":  {"name": "S3: RSI Mean Rev Long (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S4":  {"name": "S4: RSI Mean Rev Long (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S5":  {"name": "S5: RSI Mean Rev Short (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S6":  {"name": "S6: RSI Mean Rev Short (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S7":  {"name": "S7: SMA 50/200 Golden Cross (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S8":  {"name": "S8: SMA 50/200 Golden Cross (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S9":  {"name": "S9: MACD Line Crossover (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S10": {"name": "S10: MACD Line Crossover (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S11": {"name": "S11: Bollinger Bands Rev (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S12": {"name": "S12: Bollinger Bands Rev (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S13": {"name": "S13: Volatility Breakout (MU)", "type": "Rule-Based", "underlying": "MU"},
    "S14": {"name": "S14: Volatility Breakout (MUU)", "type": "Rule-Based", "underlying": "MUU"},
    "S15": {"name": "S15: Earnings Proximity Play (MU)", "type": "Earnings", "underlying": "MU"},
    "S16": {"name": "S16: Earnings Proximity Play (MUU)", "type": "Earnings", "underlying": "MUU"},
    "S17": {"name": "S17: Options Wheel Strategy (MU)", "type": "Options", "underlying": "MU"},
    "S18": {"name": "S18: Options Wheel Strategy (MUU)", "type": "Options", "underlying": "MUU"},
    "S19": {"name": "S19: FinRL PPO Agent (MU)", "type": "RL Agent", "underlying": "MU"},
    "S20": {"name": "S20: FinRL SAC Agent (MU)", "type": "RL Agent", "underlying": "MU"}
}

results_out = {}
bh_mu_return = ((portfolios["S1"][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print(f"\n| ID | Strategy Name | Return (%) | Sharpe | Max DD (%) | Beat B&H? |")
print(f"|---|---|---|---|---|---|")

for key, p_values in portfolios.items():
    meta = strategy_metadata[key]
    initial = p_values[0]
    final = p_values[-1]
    ret_pct = ((final - initial) / initial) * 100
    
    # Daily returns
    p_returns = pd.Series(p_values).pct_change().dropna()
    if p_returns.std() > 0:
        sharpe = (p_returns.mean() / p_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Drawdown
    peak = pd.Series(p_values).cummax()
    drawdowns = (pd.Series(p_values) - peak) / peak
    max_dd = drawdowns.min() * 100
    
    beat_bh = "Yes" if ret_pct > bh_mu_return else "No"
    
    results_out[key] = {
        "name": meta["name"],
        "type": meta["type"],
        "underlying": meta["underlying"],
        "dates": test_dates,
        "portfolio_values": [round(v, 2) for v in p_values],
        "final_value": round(final, 2),
        "return_pct": round(ret_pct, 2),
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(max_dd), 2),
        "beat_bh": beat_bh
    }
    
    print(f"| {key} | {meta['name']} | {ret_pct:+.1f}% | {sharpe:.3f} | {max_dd:.1f}% | {beat_bh} |")

# Save to JSON
os.makedirs("results", exist_ok=True)
with open("results/mu_multi_strategy_results.json", "w") as f:
    json.dump(results_out, f, indent=2)

print("\nSaved multi-strategy results to results/mu_multi_strategy_results.json")
