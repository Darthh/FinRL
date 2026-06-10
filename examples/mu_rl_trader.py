"""
mu_rl_trader.py  —  Use Case 1
Single-stock MU RL trading agent trained on GPU (RTX 5070 Ti).

Trains a SAC agent on Micron Technology (MU) data from 2010-2024,
then backtests on 2025+. Captures full DRAM/NAND supply cycles.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

import torch
print(f"[GPU] Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Config ──────────────────────────────────────────────────────────────────────
TICKER = "MU"
TRAIN_START = "2010-01-01"   # captures 2012 trough, 2018 peak, 2022 crash
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-06-01"

# MU-specific indicators — added RSI and earnings-adjacent signals
MU_INDICATORS = [
    "macd",        # trend following
    "rsi_14",      # momentum / overbought-oversold
    "rsi_30",      # longer-term momentum
    "boll_ub",     # upper Bollinger band (resistance)
    "boll_lb",     # lower Bollinger band (support)
    "cci_30",      # commodity channel index
    "dx_30",       # directional movement (trend strength)
    "close_30_sma",# medium-term trend
    "close_60_sma",# longer-term trend
    "close_5_sma", # short-term momentum
    "volume_delta",# unusual volume (earnings signals often show here)
]

# ── Env settings tuned for MU ───────────────────────────────────────────────────
ENV_KWARGS = {
    "hmax": 1000,                # hold up to 1000 shares (conviction sizing)
    "initial_amount": 100_000,   # $100K starting capital
    "buy_cost_pct":  [0.001],    # 0.1% commission
    "sell_cost_pct": [0.001],
    "reward_scaling": 1e-4,
    "num_stock_shares": [0],
}

# MU is volatile — higher turbulence threshold than DOW-30
TURBULENCE_THRESHOLD = 150

SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 500_000,   # large replay buffer captures cycles
    "learning_rate": 3e-4,
    "learning_starts": 5_000, # warm-up period
    "ent_coef": "auto",       # auto-tune for volatile stock
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
}

TOTAL_TIMESTEPS = 500_000   # ~30 min on RTX 5070 Ti. Use 2M for production.

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR, "results/mu"])

# ── Step 1: Download MU data ─────────────────────────────────────────────────────
print(f"\n[1/5] Downloading {TICKER} data ({TRAIN_START} → {TEST_END})...")
downloader = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TEST_END,
    ticker_list=[TICKER],
)
raw_df = downloader.fetch_data()
print(f"      Downloaded {len(raw_df)} rows")

# ── Step 2: Feature engineering ──────────────────────────────────────────────────
print(f"\n[2/5] Engineering features...")

# Use only standard INDICATORS to avoid stockstats compatibility issues
INDICATORS_TO_USE = [i for i in MU_INDICATORS if i in INDICATORS + [
    "rsi_14", "close_5_sma", "volume_delta"
]]
INDICATORS_TO_USE = INDICATORS  # fallback to standard set if custom ones fail

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS_TO_USE,
    use_turbulence=True,
    user_defined_feature=False,
)
processed = fe.preprocess_data(raw_df)
print(f"      Features: {list(processed.columns)}")

# ── Step 3: Split train/test ──────────────────────────────────────────────────────
print(f"\n[3/5] Splitting train/test...")
train_df = data_split(processed, TRAIN_START, TRAIN_END)
test_df  = data_split(processed, TEST_START,  TEST_END)
print(f"      Train: {len(train_df)} rows | Test: {len(test_df)} rows")

stock_dim   = len(train_df.tic.unique())  # = 1 for single stock
state_space = 1 + 2 * stock_dim + len(INDICATORS_TO_USE) * stock_dim
print(f"      State space: {state_space}")

env_kwargs_full = {
    **ENV_KWARGS,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS_TO_USE,
    "action_space": stock_dim,
}

# ── Step 4: Train ─────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training SAC on {TICKER} for {TOTAL_TIMESTEPS:,} timesteps...")
print(f"      GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

e_train = StockTradingEnv(df=train_df, **env_kwargs_full)
env_train, _ = e_train.get_sb_env()

agent = DRLAgent(env=env_train)
logger_sac = configure("results/mu/sac", ["stdout", "csv", "tensorboard"])
model = agent.get_model("sac", model_kwargs=SAC_PARAMS)
model.set_logger(logger_sac)

trained_model = agent.train_model(
    model=model,
    tb_log_name="mu_sac",
    total_timesteps=TOTAL_TIMESTEPS,
)
trained_model.save(f"{TRAINED_MODEL_DIR}/mu_sac")
print(f"      Model saved to {TRAINED_MODEL_DIR}/mu_sac.zip")

# ── Step 5: Backtest ──────────────────────────────────────────────────────────────
print(f"\n[5/5] Backtesting on {TEST_START} → {TEST_END}...")

e_test = StockTradingEnv(
    df=test_df,
    turbulence_threshold=TURBULENCE_THRESHOLD,
    **env_kwargs_full,
)
df_account, df_actions = DRLAgent.DRL_prediction(
    model=trained_model, environment=e_test
)

# ── Results ────────────────────────────────────────────────────────────────────────
initial  = df_account["account_value"].iloc[0]
final    = df_account["account_value"].iloc[-1]
ret_pct  = (final - initial) / initial * 100
daily_r  = df_account["account_value"].pct_change().dropna()
sharpe   = daily_r.mean() / daily_r.std() * np.sqrt(252)
peak     = df_account["account_value"].cummax()
max_dd   = ((df_account["account_value"] - peak) / peak).min() * 100

print("\n" + "="*50)
print(f"  MU RL TRADING AGENT — BACKTEST RESULTS")
print("="*50)
print(f"  Initial capital : ${initial:>12,.2f}")
print(f"  Final portfolio : ${final:>12,.2f}")
print(f"  Total return    : {ret_pct:>+11.2f}%")
print(f"  Sharpe ratio    : {sharpe:>12.3f}")
print(f"  Max drawdown    : {max_dd:>+11.2f}%")
print("="*50)

# Compare to buy-and-hold MU
bah_ret = (test_df[test_df.tic=="MU"]["close"].iloc[-1] /
           test_df[test_df.tic=="MU"]["close"].iloc[0] - 1) * 100
print(f"  Buy-and-hold MU : {bah_ret:>+11.2f}%")
print(f"  Alpha vs B&H    : {ret_pct - bah_ret:>+11.2f}%")
print("="*50)

# Save results
df_account.to_csv("results/mu_backtest_account.csv", index=False)
df_actions.to_csv("results/mu_backtest_actions.csv", index=False)
print("\nResults saved to results/mu_backtest_*.csv")
print("Run: tensorboard --logdir results/mu/sac  to see training curves")
