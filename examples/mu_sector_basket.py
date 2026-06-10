"""
mu_sector_basket.py  —  Use Case 2
MU + Semiconductor Sector Basket RL Agent

The agent allocates capital across [MU, NVDA, AMD, INTC, SOXX],
learning sector rotation dynamics. NVDA/AMD tend to lead MU by
2-4 weeks in semiconductor cycles — the agent learns this.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
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
# The semiconductor basket
# MU = primary target
# NVDA = leading indicator (AI/HBM demand, leads MU by weeks)
# AMD = correlated competitor
# INTC = lagging indicator / mean reversion signal
# SOXX = ETF benchmark / sector health
TICKERS = ["MU", "NVDA", "AMD", "INTC", "SOXX"]

TRAIN_START = "2015-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-06-01"

# Portfolio config — allocate across 5 securities
INITIAL_CAPITAL = 500_000   # $500K across 5 stocks
HMAX = 200                  # max 200 shares per stock

# Off-policy algo works best for portfolio optimization
SAC_PARAMS = {
    "batch_size": 512,
    "buffer_size": 1_000_000,
    "learning_rate": 1e-4,
    "learning_starts": 10_000,
    "ent_coef": "auto",
    "gamma": 0.99,
}

TOTAL_TIMESTEPS = 1_000_000  # 1M steps, ~45 min on RTX 5070 Ti

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR, "results/mu_basket"])

# ── Download ─────────────────────────────────────────────────────────────────────
print(f"\n[1/5] Downloading {TICKERS} data...")
downloader = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TEST_END,
    ticker_list=TICKERS,
)
raw_df = downloader.fetch_data()
print(f"      Downloaded {len(raw_df)} rows for {len(TICKERS)} tickers")

# ── Feature engineering ──────────────────────────────────────────────────────────
print(f"\n[2/5] Engineering features...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    user_defined_feature=False,
)
processed = fe.preprocess_data(raw_df)

# ── Split ────────────────────────────────────────────────────────────────────────
train_df = data_split(processed, TRAIN_START, TRAIN_END)
test_df  = data_split(processed, TEST_START,  TEST_END)

stock_dim   = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
print(f"      Stocks: {stock_dim}, State space: {state_space}")

env_kwargs = {
    "hmax": HMAX,
    "initial_amount": INITIAL_CAPITAL,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct":  [0.001] * stock_dim,
    "sell_cost_pct": [0.001] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
}

# ── Train ────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training SAC basket agent for {TOTAL_TIMESTEPS:,} timesteps...")

e_train = StockTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train.get_sb_env()

agent = DRLAgent(env=env_train)
logger = configure("results/mu_basket/sac", ["stdout", "csv", "tensorboard"])
model = agent.get_model("sac", model_kwargs=SAC_PARAMS)
model.set_logger(logger)

trained = agent.train_model(
    model=model,
    tb_log_name="mu_basket_sac",
    total_timesteps=TOTAL_TIMESTEPS,
)
trained.save(f"{TRAINED_MODEL_DIR}/mu_basket_sac")

# ── Backtest ─────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Backtesting...")
e_test = StockTradingEnv(df=test_df, turbulence_threshold=150, **env_kwargs)
df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)

# Results
initial = df_account["account_value"].iloc[0]
final   = df_account["account_value"].iloc[-1]
ret     = (final - initial) / initial * 100
daily_r = df_account["account_value"].pct_change().dropna()
sharpe  = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0

print("\n" + "="*50)
print(f"  SEMICONDUCTOR BASKET — BACKTEST RESULTS")
print("="*50)
print(f"  Tickers : {', '.join(TICKERS)}")
print(f"  Return  : {ret:+.2f}%  |  Sharpe: {sharpe:.3f}")
print("="*50)

# Show allocation over time
print("\nPortfolio actions (last 5 dates):")
print(df_actions.tail())

df_account.to_csv("results/mu_basket_account.csv", index=False)
df_actions.to_csv("results/mu_basket_actions.csv", index=False)
print("\nSaved results to results/mu_basket_*.csv")
