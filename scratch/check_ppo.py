import os
import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# Setup parameters
TICKER = "MU"
TRAIN_START = "2015-01-01"
TRAIN_END = "2025-06-01"
TEST_START = "2025-06-01"
TEST_END = "2026-06-01"
INITIAL_CAPITAL = 100_000

# Download and preprocess
download_start = (datetime.strptime(TRAIN_START, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
df_raw = yf.download(TICKER, start=download_start, end=TEST_END)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw = df_raw.reset_index().rename(columns={
    "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adjcp", "Volume": "volume"
})
df_raw["tic"] = TICKER
df_raw["day"] = pd.to_datetime(df_raw["date"]).dt.dayofweek
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")

fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_turbulence=True, user_defined_feature=False)
df_processed = fe.preprocess_data(df_raw)
df_processed = df_processed.sort_values(["date"]).reset_index(drop=True)
test_df = data_split(df_processed, TEST_START, TEST_END)

# Environment setup
stock_dim = 1
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
env_kwargs_full = {
    "hmax": 1000,
    "initial_amount": INITIAL_CAPITAL,
    "buy_cost_pct": [0.001],
    "sell_cost_pct": [0.001],
    "reward_scaling": 1e-4,
    "num_stock_shares": [0],
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
}

# Load model
ppo_path = os.path.join(TRAINED_MODEL_DIR, "mu_ppo_multi")
print("Loading model from:", ppo_path)
model_ppo = PPO.load(ppo_path) if 'PPO' in globals() else None
# Wait, let's import PPO from stable_baselines3
from stable_baselines3 import PPO
model_ppo = PPO.load(ppo_path)

e_test = StockTradingEnv(df=test_df, turbulence_threshold=150, **env_kwargs_full)
df_account, df_actions = DRLAgent.DRL_prediction(model=model_ppo, environment=e_test)

print("Account value head:")
print(df_account.head(10))
print("Actions head:")
print(df_actions.head(10))
print("Actions unique values:")
print(df_actions.value_counts())
