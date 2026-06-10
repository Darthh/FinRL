# -*- coding: utf-8 -*-
"""
run_backtest.py
Runs backtests on all 5 trained FinRL agents and exports results
to backtest_results.json for the dashboard.
"""
from __future__ import annotations
import json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

print("=" * 60)
print("  FinRL Backtest - RTX 5070 Ti")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
train = pd.read_csv("train_data.csv").set_index("Unnamed: 0")
trade = pd.read_csv("trade_data.csv").set_index("Unnamed: 0")
train.index.names = trade.index.names = [""]

stock_dim   = len(trade.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
print(f"Stock Dimension: {stock_dim}, State Space: {state_space}")

env_kwargs = dict(
    hmax=100, initial_amount=1_000_000,
    num_stock_shares=[0] * stock_dim,
    buy_cost_pct=[0.001] * stock_dim,
    sell_cost_pct=[0.001] * stock_dim,
    state_space=state_space, stock_dim=stock_dim,
    tech_indicator_list=INDICATORS,
    action_space=stock_dim, reward_scaling=1e-4,
)

# ── Load agents ────────────────────────────────────────────────
MODELS = {
    "A2C":  (A2C,  "agent_a2c"),
    "DDPG": (DDPG, "agent_ddpg"),
    "PPO":  (PPO,  "agent_ppo"),
    "TD3":  (TD3,  "agent_td3"),
    "SAC":  (SAC,  "agent_sac"),
}

results = {}

for name, (cls, fname) in MODELS.items():
    print(f"\n[{name}] Loading & backtesting...")
    try:
        model = cls.load(f"{TRAINED_MODEL_DIR}/{fname}")
        env   = StockTradingEnv(df=trade, turbulence_threshold=70,
                                risk_indicator_col="vix", **env_kwargs)
        df_account, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)

        # Build timeseries
        dates  = df_account.iloc[:, 0].tolist()
        values = df_account["account_value"].tolist()

        initial = values[0]
        final   = values[-1]
        ret_pct = (final - initial) / initial * 100

        # Sharpe (daily returns)
        import numpy as np
        daily_ret = np.diff(values) / values[:-1]
        sharpe = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * np.sqrt(252)

        # Max drawdown
        peak = np.maximum.accumulate(values)
        dd   = (np.array(values) - peak) / peak
        max_dd = float(dd.min() * 100)

        results[name] = {
            "dates":      dates,
            "values":     values,
            "initial":    initial,
            "final":      round(final, 2),
            "return_pct": round(ret_pct, 2),
            "sharpe":     round(float(sharpe), 3),
            "max_dd":     round(max_dd, 2),
        }
        print(f"  [OK] Return: {ret_pct:+.1f}%  Sharpe: {sharpe:.3f}  MaxDD: {max_dd:.1f}%")
    except Exception as e:
        print(f"  [ERR] {e}")
        results[name] = {"error": str(e)}

# ── Save JSON ──────────────────────────────────────────────────
with open("backtest_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  Results saved to backtest_results.json")
print("  Open dashboard.html in your browser to visualize!")
print("=" * 60)
