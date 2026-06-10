"""
mu_hybrid_lstm_rl.py  —  Use Case 5 (Most Institutional)
Hybrid LSTM Price Predictor + RL Position Sizer for $MU

Stage 1: LSTM predicts MU's next-day direction (up/down) + magnitude
Stage 2: RL agent uses LSTM signal as extra state feature for position sizing

This is the architecture used by institutional quant funds.
GPU (RTX 5070 Ti) accelerates both the LSTM training and RL training.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Config ───────────────────────────────────────────────────────────────────────
TICKER      = "MU"
TRAIN_START = "2010-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-06-01"

LSTM_SEQ_LEN   = 60   # 60-day lookback window for LSTM (captures a quarter)
LSTM_HIDDEN    = 128
LSTM_LAYERS    = 2
LSTM_EPOCHS    = 50
LSTM_LR        = 1e-3
LSTM_BATCH     = 64

RL_TIMESTEPS   = 500_000

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR, "results/mu_hybrid"])

# ═══════════════════════════════════════════════════════════════
# STAGE 1: LSTM PRICE DIRECTION PREDICTOR
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  STAGE 1: LSTM Training")
print("="*60)

# ── Download raw OHLCV ───────────────────────────────────────────
print(f"\n[1/3] Downloading {TICKER}...")
dl = YahooDownloader(start_date=TRAIN_START, end_date=TEST_END, ticker_list=[TICKER])
raw = dl.fetch_data()
mu_ohlcv = raw[raw.tic == TICKER].sort_values("date").reset_index(drop=True)

# Features for LSTM: OHLCV + some derived
mu_ohlcv["return_1d"]  = mu_ohlcv["close"].pct_change()
mu_ohlcv["return_5d"]  = mu_ohlcv["close"].pct_change(5)
mu_ohlcv["return_20d"] = mu_ohlcv["close"].pct_change(20)
mu_ohlcv["vol_ratio"]  = mu_ohlcv["volume"] / mu_ohlcv["volume"].rolling(20).mean()
mu_ohlcv["hl_ratio"]   = (mu_ohlcv["high"] - mu_ohlcv["low"]) / mu_ohlcv["close"]
mu_ohlcv = mu_ohlcv.dropna()

LSTM_FEATURES = ["close", "volume", "return_1d", "return_5d", "return_20d", "vol_ratio", "hl_ratio"]
TARGET = "return_1d"  # predict next-day return direction + magnitude

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_raw = mu_ohlcv[LSTM_FEATURES].values
y_raw = mu_ohlcv[TARGET].shift(-1).fillna(0).values.reshape(-1, 1)  # next day

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# ── Build sequences ──────────────────────────────────────────────
def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, LSTM_SEQ_LEN)

# Train/test split for LSTM
split_idx = int(len(X_seq) * 0.8)
X_train_l, X_test_l = X_seq[:split_idx], X_seq[split_idx:]
y_train_l, y_test_l = y_seq[:split_idx], y_seq[split_idx:]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(DEVICE)
        self.y = torch.FloatTensor(y).to(DEVICE)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train_l, y_train_l), batch_size=LSTM_BATCH, shuffle=True)
test_loader  = DataLoader(SeqDataset(X_test_l,  y_test_l),  batch_size=LSTM_BATCH)

# ── LSTM model ───────────────────────────────────────────────────
class MU_LSTM(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last timestep

lstm_model = MU_LSTM(
    input_size=len(LSTM_FEATURES),
    hidden=LSTM_HIDDEN,
    layers=LSTM_LAYERS,
).to(DEVICE)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LSTM_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
criterion = nn.MSELoss()

# ── Train LSTM ───────────────────────────────────────────────────
print(f"[2/3] Training LSTM on {TICKER} ({LSTM_EPOCHS} epochs, {DEVICE})...")
best_val_loss = float("inf")

for epoch in range(LSTM_EPOCHS):
    lstm_model.train()
    train_loss = 0
    for xb, yb in train_loader:
        pred = lstm_model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()

    lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = lstm_model(xb)
            val_loss += criterion(pred, yb).item()

    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(lstm_model.state_dict(), f"{TRAINED_MODEL_DIR}/mu_lstm_best.pt")

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:>3}/{LSTM_EPOCHS} | Train: {train_loss/len(train_loader):.5f} | Val: {val_loss:.5f}")

print(f"  Best val loss: {best_val_loss:.5f} | Model saved")

# ── Generate LSTM signals for entire dataset ─────────────────────
print("[3/3] Generating LSTM signals...")
lstm_model.load_state_dict(torch.load(f"{TRAINED_MODEL_DIR}/mu_lstm_best.pt", map_location=DEVICE))
lstm_model.eval()

all_preds = []
with torch.no_grad():
    for i in range(0, len(X_seq), 256):
        batch = torch.FloatTensor(X_seq[i:i+256]).to(DEVICE)
        pred  = lstm_model(batch).cpu().numpy()
        all_preds.extend(scaler_y.inverse_transform(pred).flatten())

# Align signals with original dataframe (offset by seq_len + 1 for next-day pred)
signal_series = pd.Series(
    [0.0] * (LSTM_SEQ_LEN + 1) + all_preds,
    index=range(len(mu_ohlcv))
)
mu_ohlcv["lstm_signal"] = signal_series.values[:len(mu_ohlcv)]
mu_ohlcv["lstm_direction"] = (mu_ohlcv["lstm_signal"] > 0).astype(float)

print(f"  LSTM signal coverage: {(mu_ohlcv['lstm_signal'] != 0).sum()} / {len(mu_ohlcv)} days")
print(f"  Directional accuracy: {(np.sign(mu_ohlcv['lstm_signal'][LSTM_SEQ_LEN+1:]) == np.sign(mu_ohlcv['return_1d'][LSTM_SEQ_LEN+1:])).mean():.1%}")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: RL AGENT WITH LSTM SIGNAL AS EXTRA STATE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  STAGE 2: RL Agent with LSTM Signal")
print("="*60)

# Download + engineer features for RL
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
)
processed = fe.preprocess_data(raw)

# Merge LSTM signal into processed DF
mu_signals = mu_ohlcv[["date", "lstm_signal", "lstm_direction"]].copy()
mu_signals["tic"] = TICKER
processed = pd.merge(processed, mu_signals, on=["date", "tic"], how="left").fillna(0)

# Add lstm_signal as a custom "indicator" the env will see in its state
INDICATORS_WITH_LSTM = INDICATORS + ["lstm_signal", "lstm_direction"]

train_df = data_split(processed, TRAIN_START, TRAIN_END)
test_df  = data_split(processed, TEST_START,  TEST_END)

stock_dim   = 1
state_space = 1 + 2 * stock_dim + len(INDICATORS_WITH_LSTM) * stock_dim

env_kwargs = {
    "hmax": 1000,
    "initial_amount": 100_000,
    "num_stock_shares": [0],
    "buy_cost_pct":  [0.001],
    "sell_cost_pct": [0.001],
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS_WITH_LSTM,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
}

print(f"\nTraining RL agent for {RL_TIMESTEPS:,} timesteps with LSTM signals...")
e_train = StockTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train.get_sb_env()

from stable_baselines3.common.logger import configure as sb3_configure
agent = DRLAgent(env=env_train)
logger = sb3_configure("results/mu_hybrid/sac", ["stdout", "csv", "tensorboard"])
model  = agent.get_model("sac", model_kwargs={
    "batch_size": 256, "buffer_size": 500_000,
    "learning_rate": 3e-4, "learning_starts": 5_000, "ent_coef": "auto",
})
model.set_logger(logger)

trained = agent.train_model(model=model, tb_log_name="mu_hybrid", total_timesteps=RL_TIMESTEPS)
trained.save(f"{TRAINED_MODEL_DIR}/mu_hybrid_sac")

# ── Backtest ─────────────────────────────────────────────────────
print("\nBacktesting hybrid model...")
e_test = StockTradingEnv(df=test_df, turbulence_threshold=150, **env_kwargs)
df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)

initial = df_account["account_value"].iloc[0]
final   = df_account["account_value"].iloc[-1]
ret     = (final - initial) / initial * 100
daily_r = df_account["account_value"].pct_change().dropna()
sharpe  = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0

print("\n" + "="*60)
print("  HYBRID LSTM + RL — BACKTEST RESULTS")
print("="*60)
print(f"  LSTM directional accuracy : see above")
print(f"  RL + LSTM return          : {ret:+.2f}%")
print(f"  Sharpe ratio              : {sharpe:.3f}")
print("="*60)

df_account.to_csv("results/mu_hybrid_account.csv", index=False)
print("\nSaved to results/mu_hybrid_account.csv")
print("Tensorboard: tensorboard --logdir results/mu_hybrid/sac")
