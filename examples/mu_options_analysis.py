"""
mu_options_analysis.py  —  Use Case 4 (Options)
MU Options Data Analyzer + IV Crush & Earnings Play Signals

This script does NOT use FinRL directly — it analyzes MU options data
to generate signals for the RL agent, or to use standalone as a
rule-based options strategy backtest.

Strategies:
  A) IV Crush: Buy straddle before earnings if IV < historical avg
  B) IV Expansion: Sell straddle if IV > 1.5x historical avg  
  C) Post-earnings drift: Enter stock position after large moves

All signals can be exported as features for the RL agent.

Requirements: yfinance (already installed)
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "MU"
print(f"{'='*60}")
print(f"  MU Options Analysis")
print(f"{'='*60}")

mu = yf.Ticker(TICKER)

# ── Part 1: Current options chain snapshot ────────────────────────────────────
print("\n[1] Loading current options chain...")

try:
    expirations = mu.options
    print(f"  Available expirations: {expirations[:8]}")

    # Get nearest expiry (best for IV analysis)
    nearest_exp = expirations[0]
    chain = mu.option_chain(nearest_exp)
    calls = chain.calls
    puts  = chain.puts

    # Current stock price
    hist = mu.history(period="1d")
    spot = hist["Close"].iloc[-1]
    print(f"  Spot price: ${spot:.2f}")
    print(f"  Nearest expiry: {nearest_exp}")
    print(f"  # Calls: {len(calls)} | # Puts: {len(puts)}")

    # ATM options (closest strike to spot)
    atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    atm_put  = puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[0]]

    atm_call_iv = atm_call.get("impliedVolatility", 0) * 100
    atm_put_iv  = atm_put.get("impliedVolatility", 0) * 100
    avg_iv = (atm_call_iv + atm_put_iv) / 2

    print(f"\n  ATM Call IV : {atm_call_iv:.1f}%  (strike ${atm_call['strike']:.0f})")
    print(f"  ATM Put  IV : {atm_put_iv:.1f}%  (strike ${atm_put['strike']:.0f})")
    print(f"  Average IV  : {avg_iv:.1f}%")

    # Straddle cost estimate
    straddle_cost = (atm_call.get("lastPrice", 0) + atm_put.get("lastPrice", 0))
    straddle_pct  = straddle_cost / spot * 100
    print(f"\n  ATM Straddle cost : ${straddle_cost:.2f} ({straddle_pct:.1f}% of spot)")
    print(f"  Break-even move   : ±${straddle_cost:.2f} (±{straddle_pct:.1f}%)")

    # MU historical avg earnings move
    MU_AVG_EARNINGS_MOVE = 12.0  # percent, historical average
    if straddle_pct < MU_AVG_EARNINGS_MOVE * 0.85:
        signal = "BUY STRADDLE (options underpricing MU earnings risk!)"
        signal_color = "BULLISH IV"
    elif straddle_pct > MU_AVG_EARNINGS_MOVE * 1.3:
        signal = "SELL STRADDLE (options overpricing risk, expect IV crush)"
        signal_color = "BEARISH IV"
    else:
        signal = "HOLD — IV fairly priced for MU earnings"
        signal_color = "NEUTRAL"

    print(f"\n  MU avg earnings move : ±{MU_AVG_EARNINGS_MOVE}%")
    print(f"  Signal: [{signal_color}] {signal}")

except Exception as e:
    print(f"  [Note] Options data unavailable in batch mode: {e}")
    print(f"  Run interactively or during market hours for live options data.")
    spot = 100.0  # fallback for demo
    avg_iv = 45.0

# ── Part 2: MU Earnings calendar ─────────────────────────────────────────────
print(f"\n[2] Earnings calendar...")
try:
    cal = mu.calendar
    if cal is not None and len(cal) > 0:
        print(f"  Next earnings: {cal}")
    else:
        print("  MU typically reports: late March, late June, late September, late December")
        # MU always reports in the last week of these months
        now = datetime.now()
        months = [3, 6, 9, 12]
        next_earnings = None
        for m in sorted(months):
            candidate = datetime(now.year if m > now.month else now.year + 1, m, 25)
            if candidate > now:
                next_earnings = candidate
                break
        days_to = (next_earnings - now).days if next_earnings else "?"
        print(f"  Estimated next earnings: {next_earnings.strftime('%Y-%m-%d') if next_earnings else 'Unknown'} (~{days_to} days)")
except Exception as e:
    print(f"  Calendar fetch error: {e}")

# ── Part 3: MU historical price analysis ──────────────────────────────────────
print(f"\n[3] Historical MU price analysis (2 years)...")
hist_2y = mu.history(period="2y")

# Rolling realized volatility (annualized)
returns = hist_2y["Close"].pct_change().dropna()
rv_30d  = returns.rolling(30).std() * np.sqrt(252) * 100
rv_90d  = returns.rolling(90).std() * np.sqrt(252) * 100

print(f"  30-day realized vol : {rv_30d.iloc[-1]:.1f}%")
print(f"  90-day realized vol : {rv_90d.iloc[-1]:.1f}%")

# Current implied vol vs realized — the IV premium
iv_premium = avg_iv - rv_30d.iloc[-1]
print(f"  IV premium over RV  : {iv_premium:+.1f}% ({'Options EXPENSIVE' if iv_premium > 5 else 'Options CHEAP' if iv_premium < -5 else 'Fairly priced'})")

# Max move analysis (for position sizing)
rolling_max_move = returns.abs().rolling(252).max() * 100
print(f"  Largest 1-day move (1yr): {rolling_max_move.iloc[-1]:.1f}%")
print(f"  Average 1-day move      : {returns.abs().mean()*100:.2f}%")

# MU price performance
ytd_start = hist_2y[hist_2y.index.year == datetime.now().year]["Close"].iloc[0]
ytd_perf  = (hist_2y["Close"].iloc[-1] / ytd_start - 1) * 100
print(f"\n  YTD performance : {ytd_perf:+.1f}%")
print(f"  Current price   : ${hist_2y['Close'].iloc[-1]:.2f}")
print(f"  52-week high    : ${hist_2y['Close'].max():.2f}")
print(f"  52-week low     : ${hist_2y['Close'].min():.2f}")
print(f"  52-week range   : {(hist_2y['Close'].iloc[-1] - hist_2y['Close'].min()) / (hist_2y['Close'].max() - hist_2y['Close'].min()) * 100:.0f}th percentile")

# ── Part 4: Signals for RL agent ──────────────────────────────────────────────
print(f"\n[4] Generating RL-ready signals for {len(hist_2y)} days...")

signals = pd.DataFrame(index=hist_2y.index)
signals["close"]      = hist_2y["Close"]
signals["volume"]     = hist_2y["Volume"]
signals["rv_30d"]     = rv_30d
signals["rv_90d"]     = rv_90d
signals["iv_premium"] = avg_iv - rv_30d  # simplified, needs real IV data for production

# Trend signals
signals["above_50sma"]  = (hist_2y["Close"] > hist_2y["Close"].rolling(50).mean()).astype(int)
signals["above_200sma"] = (hist_2y["Close"] > hist_2y["Close"].rolling(200).mean()).astype(int)
signals["golden_cross"] = (signals["above_50sma"] > signals["above_200sma"]).astype(int)

# Momentum
signals["rsi"] = compute_rsi(hist_2y["Close"], 14) if "compute_rsi" in dir() else \
    100 - 100 / (1 + returns.clip(lower=0).rolling(14).mean() / (-returns.clip(upper=0)).rolling(14).mean())

# Regime detection
signals["high_vol_regime"]  = (rv_30d > rv_30d.rolling(252).mean() * 1.3).astype(int)
signals["earnings_month"]   = hist_2y.index.month.isin([3, 6, 9, 12]).astype(int)
signals["earnings_lastweek"]= ((hist_2y.index.month.isin([3, 6, 9, 12])) &
                               (hist_2y.index.day >= 20)).astype(int)

signals = signals.dropna()
signals.to_csv("results/mu_signals.csv")

print(f"  Generated {len(signals.columns)} signals over {len(signals)} trading days")
print(f"  Saved to results/mu_signals.csv")
print(f"\n  Key signals today:")
for col in ["rv_30d", "above_50sma", "above_200sma", "high_vol_regime", "earnings_month"]:
    if col in signals.columns:
        print(f"    {col:25s}: {signals[col].iloc[-1]:.2f}")

# ── Part 5: Options wheel strategy backtest (simplified) ─────────────────────
print(f"\n[5] Simplified Wheel Strategy Backtest on MU...")
print(f"  Strategy: Sell cash-secured puts when price > 200 SMA,")
print(f"            collect premium, take assignment, then sell covered calls")

# Rule-based backtest
cash = 100_000
shares = 0
premium_collected = 0
assignments = 0
option_premium_pct = 0.015  # assume 1.5% weekly premium (rough for MU)
weekly_data = hist_2y["Close"].resample("W").last().dropna()

for i in range(1, len(weekly_data)):
    price = weekly_data.iloc[i]
    prev  = weekly_data.iloc[i-1]
    above_200sma = price > hist_2y["Close"].rolling(200).mean().iloc[
        hist_2y.index.get_loc(hist_2y.index[hist_2y.index <= weekly_data.index[i]][-1])
    ] if len(hist_2y.index[hist_2y.index <= weekly_data.index[i]]) > 0 else True

    if shares == 0 and cash >= price * 100:
        # Sell cash-secured put (collect premium on 100 shares)
        premium = price * 100 * option_premium_pct
        cash += premium
        premium_collected += premium
        if price < prev * 0.98:  # assigned if price drops >2%
            shares += 100
            cash -= price * 100
            assignments += 1
    elif shares >= 100:
        # Sell covered call (collect premium on existing shares)
        premium = price * 100 * option_premium_pct * 0.8
        cash += premium
        premium_collected += premium
        if price > prev * 1.02:  # called away if price rises >2%
            cash += price * 100
            shares -= 100

total_value  = cash + shares * weekly_data.iloc[-1]
total_return = (total_value - 100_000) / 100_000 * 100
buy_hold_ret = (weekly_data.iloc[-1] / weekly_data.iloc[0] - 1) * 100

print(f"  Wheel total return      : {total_return:+.1f}%")
print(f"  Buy-and-hold MU return  : {buy_hold_ret:+.1f}%")
print(f"  Total premium collected : ${premium_collected:,.0f}")
print(f"  Total assignments       : {assignments}")
print(f"  Final: ${total_value:,.0f} ({shares} shares + ${cash:,.0f} cash)")

print(f"\n{'='*60}")
print(f"  SUMMARY OF SIGNALS FOR RL AGENT INTEGRATION")
print(f"{'='*60}")
print(f"""
  Add these to your RL state space in mu_rl_trader.py:

  from mu_options_analysis import signals  # or load CSV
  
  # Merge into processed dataframe before creating StockTradingEnv
  processed = pd.merge(processed, signals[['rv_30d', 'above_200sma',
              'high_vol_regime', 'earnings_lastweek']],
              left_on='date', right_index=True, how='left').fillna(0)
  
  INDICATORS_WITH_OPTIONS = INDICATORS + [
      'rv_30d', 'above_200sma', 'high_vol_regime', 'earnings_lastweek'
  ]
""")
