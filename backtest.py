#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Daily Strategy Backtest System - Institutional Enhanced Edition v4.0
===========================================================================
Core goals:
1. Match train.py feature engineering exactly.
2. Run a practical portfolio-level backtest for the future period after cutoff.
3. Improve diagnostics, transparency, and risk management.
4. Preserve a rich single-file structure similar to a production research script.

Main improvements versus the previous backtest script:
- Explicit feature synchronization with train.py metadata.
- Better data validation and richer logging.
- Top-N candidate ranking + confidence floor.
- Dynamic position sizing using signal strength and ATR.
- Portfolio exposure limits, cooldown, timeout, signal exit, trailing stop.
- Daily diagnostics and detailed trade ledger.
- Full report artifacts: equity curve, drawdown, summary CSVs and plots.
"""

# ======================================================================
# 0) UTF-8 / environment safety
# ======================================================================
import sys
import io
import os
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# ======================================================================
# 1) Imports
# ======================================================================
import json
import math
import glob
import warnings
import logging
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ======================================================================
# 2) Logging
# ======================================================================
os.makedirs('reports', exist_ok=True)
os.makedirs('plots', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print('=' * 80)
print('XGBoost Daily Strategy Backtest System - Institutional Enhanced Edition v4.0')
print('=' * 80)

# ======================================================================
# 3) Configuration
# ======================================================================
DATA_PATH = 'daily_data.csv'
MODEL_PATTERN = 'models/xgboost_daily_model_*.pkl'
METADATA_PATH = 'models/train_metadata.pkl'
CUTOFF_FILE = 'cutoff_date.txt'

INITIAL_CASH = 100000.0
COMMISSION = 0.0015
SLIPPAGE = 0.0005
MIN_CONFIDENCE = 0.55
SOFT_MIN_PROB = 0.50
EXIT_THRESHOLD = 0.45
TOP_N_PER_DAY = 3
MAX_POSITIONS = 5
BASE_POSITION_SIZE = 0.10
MAX_SINGLE_POSITION = 0.20
MAX_PORTFOLIO_EXPOSURE = 1.00
MIN_TRADE_INTERVAL = 2
MAX_HOLDING_DAYS = 15
STOP_LOSS_PCT = -0.12
TRAILING_STOP_PCT = 0.08
ATR_STOP_MULTIPLIER = 1.0
MIN_SAMPLES_PER_SYMBOL = 80

REQUIRED_COLS = ['date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']

# ======================================================================
# 4) Utility helpers
# ======================================================================
def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def ensure_required_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def calculate_sharpe(returns):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def calculate_sortino(returns):
    returns = pd.Series(returns).dropna()
    downside = returns[returns < 0]
    if len(returns) == 0 or len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(252))


def calculate_max_drawdown(equity):
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return float(dd.min())


def calculate_cagr(equity):
    equity = pd.Series(equity).dropna()
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / 252.0
    if years <= 0:
        return 0.0
    try:
        return float((1.0 + total_return) ** (1.0 / years) - 1.0)
    except Exception:
        return 0.0


def calculate_calmar(cagr, max_drawdown):
    if max_drawdown == 0:
        return 0.0
    return float(cagr / abs(max_drawdown))


def load_cutoff_date():
    if not os.path.exists(CUTOFF_FILE):
        raise FileNotFoundError(f'Cannot find {CUTOFF_FILE}. Run train.py first.')
    with open(CUTOFF_FILE, 'r', encoding='utf-8') as f:
        return pd.to_datetime(f.read().strip())


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f'Cannot find {METADATA_PATH}. Run train.py first.')
    return joblib.load(METADATA_PATH)


# ======================================================================
# 5) Data loading
# ======================================================================
def load_daily_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot find {path}')
    df = pd.read_csv(path, parse_dates=['date'])
    ensure_required_columns(df)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.drop_duplicates(subset=['symbol', 'date']).copy()
    df = df.dropna(subset=['date', 'symbol', 'Close']).copy()
    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    return df


# ======================================================================
# 6) Feature engineering contract
# IMPORTANT: must match train.py create_daily_features()
# ======================================================================
def create_daily_features(df):
    df = df.copy().sort_values(['symbol', 'date']).reset_index(drop=True)
    symbols = df['symbol'].unique()
    feature_cols = []

    ma_windows = [5, 10, 20, 50, 100]
    vol_windows = [5, 10, 20, 40]
    ret_windows = [1, 3, 5, 10, 20]
    bb_window = 20

    init_cols = []
    for w in ma_windows:
        init_cols += [
            f'MA_{w}',
            f'MA_deviation_{w}',
            f'MA_slope_{w}',
            f'price_to_MA_{w}',
        ]

    init_cols += [
        f'BB_middle_{bb_window}',
        f'BB_std_{bb_window}',
        f'BB_upper_{bb_window}',
        f'BB_lower_{bb_window}',
        f'BB_position_{bb_window}',
        f'BB_distance_std_{bb_window}',
        f'BB_width_{bb_window}',
    ]

    for w in vol_windows:
        init_cols += [
            f'volatility_{w}',
            f'volatility_ratio_{w}',
            f'volume_MA_{w}',
        ]

    for w in ret_windows:
        init_cols += [
            f'return_{w}',
            f'momentum_{w}',
        ]

    init_cols += [
        'return',
        'overnight_return',
        'intraday_return',
        'gap_to_prev_close',
        'high_low_range',
        'candle_body',
        'upper_shadow',
        'lower_shadow',
        'volume_ratio',
        'volume_change',
        'volume_zscore_20',
        'RSI_14',
        'RSI_overbought',
        'RSI_oversold',
        'MACD',
        'MACD_signal',
        'MACD_hist',
        'stoch_k',
        'stoch_d',
        'adx',
        'obv',
        'atr_14',
        'atr_pct',
        'donchian_high_20',
        'donchian_low_20',
        'donchian_position_20',
        'volatility_regime',
        'trend_strength',
        'range_ratio_20',
        'day_of_week',
        'month',
        'quarter',
        'month_end_flag',
    ]

    for lag in [1, 2, 3, 4, 5]:
        init_cols += [
            f'return_lag_{lag}',
            f'MA_deviation_lag_{lag}',
            f'volume_ratio_lag_{lag}',
            f'RSI_lag_{lag}',
        ]

    for col in init_cols:
        df[col] = np.nan
        if col not in feature_cols:
            feature_cols.append(col)

    for symbol in symbols:
        mask = df['symbol'] == symbol
        if mask.sum() < MIN_SAMPLES_PER_SYMBOL:
            continue

        close = df.loc[mask, 'Close']
        open_ = df.loc[mask, 'Open']
        high = df.loc[mask, 'High']
        low = df.loc[mask, 'Low']
        volume = df.loc[mask, 'Volume']

        prev_close = close.shift(1)
        returns = close.pct_change()
        df.loc[mask, 'return'] = returns
        df.loc[mask, 'overnight_return'] = safe_div(open_ - prev_close, prev_close)
        df.loc[mask, 'intraday_return'] = safe_div(close - open_, open_)
        df.loc[mask, 'gap_to_prev_close'] = safe_div(open_ - prev_close, prev_close)
        df.loc[mask, 'high_low_range'] = safe_div(high - low, close)
        df.loc[mask, 'candle_body'] = safe_div(close - open_, open_)
        df.loc[mask, 'upper_shadow'] = safe_div(high - np.maximum(open_, close), close)
        df.loc[mask, 'lower_shadow'] = safe_div(np.minimum(open_, close) - low, close)

        for w in ret_windows:
            df.loc[mask, f'return_{w}'] = close.pct_change(w)
            df.loc[mask, f'momentum_{w}'] = safe_div(close - close.shift(w), close.shift(w))

        for w in ma_windows:
            ma = close.rolling(w).mean()
            df.loc[mask, f'MA_{w}'] = ma
            df.loc[mask, f'MA_deviation_{w}'] = safe_div(close - ma, ma) * 100.0
            df.loc[mask, f'MA_slope_{w}'] = ma.pct_change()
            df.loc[mask, f'price_to_MA_{w}'] = safe_div(close, ma)

        bb_mid = close.rolling(bb_window).mean()
        bb_std = close.rolling(bb_window).std()
        bb_up = bb_mid + 2 * bb_std
        bb_lo = bb_mid - 2 * bb_std
        df.loc[mask, f'BB_middle_{bb_window}'] = bb_mid
        df.loc[mask, f'BB_std_{bb_window}'] = bb_std
        df.loc[mask, f'BB_upper_{bb_window}'] = bb_up
        df.loc[mask, f'BB_lower_{bb_window}'] = bb_lo
        df.loc[mask, f'BB_position_{bb_window}'] = safe_div(close - bb_lo, bb_up - bb_lo)
        df.loc[mask, f'BB_distance_std_{bb_window}'] = safe_div(close - bb_mid, bb_std)
        df.loc[mask, f'BB_width_{bb_window}'] = safe_div(bb_up - bb_lo, bb_mid)

        for w in vol_windows:
            vol = returns.rolling(w).std()
            df.loc[mask, f'volatility_{w}'] = vol
            df.loc[mask, f'volatility_ratio_{w}'] = safe_div(vol, vol.rolling(20).mean())
            df.loc[mask, f'volume_MA_{w}'] = volume.rolling(w).mean()

        df.loc[mask, 'volume_ratio'] = safe_div(volume, df.loc[mask, 'volume_MA_10'])
        df.loc[mask, 'volume_change'] = volume.pct_change()
        v20 = volume.rolling(20)
        df.loc[mask, 'volume_zscore_20'] = safe_div(volume - v20.mean(), v20.std())

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = safe_div(avg_gain, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df.loc[mask, 'RSI_14'] = rsi
        df.loc[mask, 'RSI_overbought'] = (rsi > 70).astype(int)
        df.loc[mask, 'RSI_oversold'] = (rsi < 30).astype(int)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        df.loc[mask, 'MACD'] = macd
        df.loc[mask, 'MACD_signal'] = macd_signal
        df.loc[mask, 'MACD_hist'] = macd_hist

        low_min = low.rolling(14).min()
        high_max = high.rolling(14).max()
        stoch_k = 100.0 * safe_div(close - low_min, high_max - low_min)
        stoch_d = stoch_k.rolling(3).mean()
        df.loc[mask, 'stoch_k'] = stoch_k
        df.loc[mask, 'stoch_d'] = stoch_d

        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        atr = pd.Series(tr.values, index=close.index).ewm(span=14, adjust=False).mean()
        df.loc[mask, 'atr_14'] = atr
        df.loc[mask, 'atr_pct'] = safe_div(atr, close)

        up_move = high.diff()
        down_move = -low.diff()
        dm_plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
        dm_minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
        di_plus = 100.0 * safe_div(dm_plus.ewm(span=14, adjust=False).mean(), atr)
        di_minus = 100.0 * safe_div(dm_minus.ewm(span=14, adjust=False).mean(), atr)
        dx = 100.0 * safe_div(abs(di_plus - di_minus), di_plus + di_minus)
        df.loc[mask, 'adx'] = dx.ewm(span=14, adjust=False).mean()

        direction = np.sign(close.diff()).fillna(0.0)
        df.loc[mask, 'obv'] = (volume * direction).cumsum()

        don_high = high.rolling(20).max()
        don_low = low.rolling(20).min()
        df.loc[mask, 'donchian_high_20'] = don_high
        df.loc[mask, 'donchian_low_20'] = don_low
        df.loc[mask, 'donchian_position_20'] = safe_div(close - don_low, don_high - don_low)

        df.loc[mask, 'volatility_regime'] = (df.loc[mask, 'volatility_5'] > df.loc[mask, 'volatility_20']).astype(float)
        df.loc[mask, 'trend_strength'] = safe_div(df.loc[mask, 'MA_20'] - df.loc[mask, 'MA_50'], df.loc[mask, 'MA_50'])
        df.loc[mask, 'range_ratio_20'] = safe_div((high - low).rolling(20).mean(), close)

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_end_flag'] = df['date'].dt.is_month_end.astype(int)

    for lag in [1, 2, 3, 4, 5]:
        df[f'return_lag_{lag}'] = df.groupby('symbol')['return'].shift(lag)
        df[f'MA_deviation_lag_{lag}'] = df.groupby('symbol')['MA_deviation_20'].shift(lag)
        df[f'volume_ratio_lag_{lag}'] = df.groupby('symbol')['volume_ratio'].shift(lag)
        df[f'RSI_lag_{lag}'] = df.groupby('symbol')['RSI_14'].shift(lag)

    feature_cols = list(dict.fromkeys(feature_cols))
    return df, feature_cols


# ======================================================================
# 7) Model loading and scoring helpers
# ======================================================================
def load_ensemble_models(pattern=MODEL_PATTERN):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No model files found: {pattern}')
    ensemble = []
    feature_sets = []
    for f in files:
        md = joblib.load(f)
        ensemble.append(md)
        feature_sets.append(tuple(md['features']))
        logger.info(f'Loaded model {os.path.basename(f)} | valid_auc={md.get("valid_auc", "N/A")}')
    if len(set(feature_sets)) != 1:
        raise ValueError('Feature mismatch across model files.')
    return ensemble


def score_row_ensemble(row_values, ensemble):
    probs = []
    for md in ensemble:
        p = md['model'].predict_proba(row_values)[0][1]
        probs.append(float(p))
    return float(np.mean(probs)), probs


def calculate_position_fraction(signal_prob, atr_pct, current_exposure):
    if np.isnan(signal_prob) or signal_prob < SOFT_MIN_PROB:
        return 0.0
    if np.isnan(atr_pct) or atr_pct <= 0:
        atr_pct = 0.02
    if current_exposure >= MAX_PORTFOLIO_EXPOSURE:
        return 0.0

    base = BASE_POSITION_SIZE
    if signal_prob >= 0.70:
        signal_multiplier = 1.60
    elif signal_prob >= 0.65:
        signal_multiplier = 1.35
    elif signal_prob >= 0.60:
        signal_multiplier = 1.15
    elif signal_prob >= 0.55:
        signal_multiplier = 1.00
    else:
        signal_multiplier = 0.75

    volatility_penalty = min(max(0.015 / atr_pct, 0.50), 1.25)
    pos_frac = base * signal_multiplier * volatility_penalty
    remaining_capacity = max(MAX_PORTFOLIO_EXPOSURE - current_exposure, 0.0)
    pos_frac = min(pos_frac, MAX_SINGLE_POSITION, remaining_capacity)
    pos_frac = max(pos_frac, 0.0)
    return float(pos_frac)


# ======================================================================
# 8) Reporting helpers
# ======================================================================
def summarize_trades(trades_df):
    if trades_df.empty:
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_return_per_trade': 0.0,
            'avg_holding_days': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
        }

    wins = trades_df.loc[trades_df['pnl'] > 0, 'pnl']
    losses = trades_df.loc[trades_df['pnl'] < 0, 'pnl']
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss_abs = float(abs(losses.mean())) if len(losses) else 0.0
    win_rate = float((trades_df['pnl'] > 0).mean())
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else 0.0
    expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss_abs

    return {
        'num_trades': int(len(trades_df)),
        'win_rate': win_rate,
        'avg_return_per_trade': float(trades_df['return_pct'].mean()),
        'avg_holding_days': float(trades_df['holding_days'].mean()),
        'avg_win_pnl': avg_win,
        'avg_loss_pnl_abs': avg_loss_abs,
        'profit_factor': profit_factor,
        'expectancy': float(expectancy),
    }


def make_plots(equity_df, trades_df):
    if equity_df.empty:
        return

    rolling_max = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(equity_df.index, equity_df['equity'], color='royalblue', lw=1.5)
    axes[0].axhline(INITIAL_CASH, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Portfolio Equity Curve')
    axes[0].set_ylabel('Equity')
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(drawdown.index, drawdown.values, 0, color='tomato', alpha=0.5)
    axes[1].set_title('Drawdown (%)')
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/backtest_equity_drawdown.png', dpi=180)
    plt.close()

    if not trades_df.empty:
        plt.figure(figsize=(10, 5))
        trades_df['return_pct'].hist(bins=30, color='slateblue', alpha=0.8)
        plt.title('Trade Return Distribution (%)')
        plt.xlabel('Return %')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots/trade_return_distribution.png', dpi=180)
        plt.close()

        exit_counts = trades_df['exit_reason'].value_counts()
        plt.figure(figsize=(8, 4))
        exit_counts.plot(kind='bar', color='darkorange')
        plt.title('Exit Reason Counts')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots/exit_reason_counts.png', dpi=180)
        plt.close()


# ======================================================================
# 9) Main backtest pipeline
# ======================================================================
def main():
    logger.info('=' * 80)
    logger.info('Starting backtest pipeline...')

    # --------------------------------------------------------------
    # Load metadata and data
    # --------------------------------------------------------------
    cutoff_date = load_cutoff_date()
    metadata = load_metadata()
    ensemble = load_ensemble_models(MODEL_PATTERN)
    expected_features = metadata['feature_cols']

    logger.info(f'Cutoff date: {cutoff_date}')
    raw_df = load_daily_data(DATA_PATH)
    logger.info(f'Loaded dataset rows: {len(raw_df)} | symbols: {raw_df["symbol"].nunique()}')

    # --------------------------------------------------------------
    # Feature generation
    # --------------------------------------------------------------
    logger.info('Creating features for backtest...')
    feat_df, feature_cols = create_daily_features(raw_df)

    if set(feature_cols) != set(expected_features):
        missing_in_backtest = sorted(set(expected_features) - set(feature_cols))
        extra_in_backtest = sorted(set(feature_cols) - set(expected_features))
        logger.warning(f'Feature mismatch detected.')
        logger.warning(f'  Missing in backtest: {missing_in_backtest[:10]}')
        logger.warning(f'  Extra in backtest  : {extra_in_backtest[:10]}')

    feature_cols = expected_features
    feat_df = feat_df[feat_df['date'] > cutoff_date].copy()
    if feat_df.empty:
        raise ValueError('No future-period data found after cutoff date.')

    pre_drop_rows = len(feat_df)
    feat_df = feat_df.dropna(subset=feature_cols).copy()
    logger.info(f'Backtest rows after NaN drop: {len(feat_df)} (dropped {pre_drop_rows - len(feat_df)})')

    # --------------------------------------------------------------
    # Precompute daily row index for faster lookup
    # --------------------------------------------------------------
    daily_groups = {d: g.copy() for d, g in feat_df.groupby('date')}
    all_dates = np.sort(feat_df['date'].unique())
    logger.info(f'Backtest dates: {len(all_dates)}')

    # --------------------------------------------------------------
    # Portfolio state
    # --------------------------------------------------------------
    cash = INITIAL_CASH
    positions = {}
    last_trade_date = {}
    equity_curve = []
    trades = []
    diagnostics_rows = []

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------
    for current_date in all_dates:
        daily = daily_groups[current_date]
        daily_by_symbol = {r['symbol']: r for _, r in daily.iterrows()}

        # ----------------------------------------------------------
        # Mark existing positions to market
        # ----------------------------------------------------------
        marked_value = 0.0
        for sym, pos in positions.items():
            if sym in daily_by_symbol:
                pos['last_price'] = float(daily_by_symbol[sym]['Close'])
            marked_value += pos['shares'] * pos['last_price']
        total_equity = cash + marked_value
        current_exposure = safe_div(marked_value, total_equity) if total_equity > 0 else 0.0

        # ----------------------------------------------------------
        # Exit logic
        # ----------------------------------------------------------
        for sym in list(positions.keys()):
            if sym not in daily_by_symbol:
                continue
            row = daily_by_symbol[sym]
            pos = positions[sym]

            close_px = float(row['Close'])
            open_px = float(row['Open']) if not pd.isna(row['Open']) else close_px
            high_px = float(row['High']) if not pd.isna(row['High']) else close_px
            low_px = float(row['Low']) if not pd.isna(row['Low']) else close_px
            atr_px = float(row['atr_14']) if not pd.isna(row['atr_14']) else close_px * 0.02
            pos['last_price'] = close_px

            row_values = row[feature_cols].values.reshape(1, -1)
            avg_prob, per_model_probs = score_row_ensemble(row_values, ensemble)

            if close_px > pos['peak_price']:
                pos['peak_price'] = close_px
                pos['trailing_stop_price'] = close_px * (1.0 - TRAILING_STOP_PCT)

            holding_days = int((current_date - pos['entry_date']).days)
            pnl_pct = safe_div(close_px - pos['entry_price'], pos['entry_price'])
            atr_stop_price = pos['entry_price'] - ATR_STOP_MULTIPLIER * pos['entry_atr']

            exit_reason = None
            if pnl_pct <= STOP_LOSS_PCT:
                exit_reason = 'fixed_stop_loss'
            elif low_px <= atr_stop_price:
                exit_reason = 'atr_stop'
            elif close_px <= pos['trailing_stop_price']:
                exit_reason = 'trailing_stop'
            elif avg_prob < EXIT_THRESHOLD:
                exit_reason = 'signal_exit'
            elif holding_days >= MAX_HOLDING_DAYS:
                exit_reason = 'time_exit'

            if exit_reason is not None:
                exit_price = close_px * (1.0 - SLIPPAGE)
                proceeds = pos['shares'] * exit_price * (1.0 - COMMISSION)
                cash += proceeds
                pnl = proceeds - pos['cost_basis']
                trades.append({
                    'symbol': sym,
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'cost_basis': pos['cost_basis'],
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'return_pct': safe_div(exit_price - pos['entry_price'], pos['entry_price']) * 100.0,
                    'holding_days': holding_days,
                    'entry_prob': pos['entry_prob'],
                    'exit_prob': avg_prob,
                    'entry_atr_pct': pos['entry_atr_pct'],
                    'exit_reason': exit_reason,
                    'peak_price': pos['peak_price'],
                    'trailing_stop_price': pos['trailing_stop_price'],
                    'atr_stop_price': atr_stop_price,
                })
                last_trade_date[sym] = current_date
                del positions[sym]

        # ----------------------------------------------------------
        # Re-mark portfolio after exits
        # ----------------------------------------------------------
        marked_value = 0.0
        for sym, pos in positions.items():
            if sym in daily_by_symbol:
                pos['last_price'] = float(daily_by_symbol[sym]['Close'])
            marked_value += pos['shares'] * pos['last_price']
        total_equity = cash + marked_value
        current_exposure = safe_div(marked_value, total_equity) if total_equity > 0 else 0.0

        # ----------------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------------
        candidates = []
        prob_rows = []
        skipped_nan = 0
        skipped_cooldown = 0
        skipped_existing = 0

        for _, row in daily.iterrows():
            sym = row['symbol']
            if sym in positions:
                skipped_existing += 1
                continue

            last_dt = last_trade_date.get(sym, pd.Timestamp('1970-01-01'))
            gap = int((current_date - last_dt).days)
            if gap < MIN_TRADE_INTERVAL:
                skipped_cooldown += 1
                continue

            row_values = row[feature_cols].values.reshape(1, -1)
            if np.any(np.isnan(row_values)):
                skipped_nan += 1
                continue

            avg_prob, per_model_probs = score_row_ensemble(row_values, ensemble)
            prob_rows.append((sym, avg_prob))

            if avg_prob >= SOFT_MIN_PROB:
                candidates.append({
                    'symbol': sym,
                    'date': current_date,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'atr_14': float(row['atr_14']) if not pd.isna(row['atr_14']) else float(row['Close']) * 0.02,
                    'atr_pct': float(row['atr_pct']) if not pd.isna(row['atr_pct']) else 0.02,
                    'prob': avg_prob,
                })

        prob_values = [p for _, p in prob_rows]
        hard_count = int(sum(p >= MIN_CONFIDENCE for p in prob_values))
        soft_count = int(sum(p >= SOFT_MIN_PROB for p in prob_values))

        diagnostics_rows.append({
            'date': current_date,
            'num_rows': int(len(daily)),
            'positions_open': int(len(positions)),
            'cash': float(cash),
            'marked_value': float(marked_value),
            'total_equity': float(total_equity),
            'current_exposure': float(current_exposure),
            'prob_min': float(np.min(prob_values)) if prob_values else np.nan,
            'prob_mean': float(np.mean(prob_values)) if prob_values else np.nan,
            'prob_max': float(np.max(prob_values)) if prob_values else np.nan,
            'hard_count': hard_count,
            'soft_count': soft_count,
            'candidate_count': int(len(candidates)),
            'skipped_nan': int(skipped_nan),
            'skipped_cooldown': int(skipped_cooldown),
            'skipped_existing': int(skipped_existing),
        })

        # ----------------------------------------------------------
        # Position entries
        # ----------------------------------------------------------
        candidates = sorted(candidates, key=lambda x: x['prob'], reverse=True)[:TOP_N_PER_DAY]

        for cand in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            if cand['prob'] < SOFT_MIN_PROB:
                continue

            marked_value = sum(pos['shares'] * pos['last_price'] for pos in positions.values())
            total_equity = cash + marked_value
            current_exposure = safe_div(marked_value, total_equity) if total_equity > 0 else 0.0
            if current_exposure >= MAX_PORTFOLIO_EXPOSURE:
                break

            position_frac = calculate_position_fraction(cand['prob'], cand['atr_pct'], current_exposure)
            if position_frac <= 0:
                continue

            desired_value = total_equity * position_frac
            if desired_value <= 0:
                continue

            entry_price = cand['open'] * (1.0 + SLIPPAGE)
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            gross_shares = int(desired_value // (entry_price * (1.0 + COMMISSION)))
            if gross_shares <= 0:
                continue

            total_cost = gross_shares * entry_price * (1.0 + COMMISSION)
            if total_cost > cash:
                gross_shares = int(cash // (entry_price * (1.0 + COMMISSION)))
                if gross_shares <= 0:
                    continue
                total_cost = gross_shares * entry_price * (1.0 + COMMISSION)

            cash -= total_cost
            positions[cand['symbol']] = {
                'symbol': cand['symbol'],
                'entry_date': current_date,
                'entry_price': entry_price,
                'shares': gross_shares,
                'cost_basis': total_cost,
                'entry_prob': cand['prob'],
                'entry_atr': cand['atr_14'],
                'entry_atr_pct': cand['atr_pct'],
                'peak_price': entry_price,
                'trailing_stop_price': entry_price * (1.0 - TRAILING_STOP_PCT),
                'last_price': entry_price,
            }
            last_trade_date[cand['symbol']] = current_date

        # ----------------------------------------------------------
        # Daily equity snapshot
        # ----------------------------------------------------------
        marked_value = sum(pos['shares'] * pos['last_price'] for pos in positions.values())
        total_equity = cash + marked_value
        equity_curve.append({
            'date': current_date,
            'cash': float(cash),
            'marked_value': float(marked_value),
            'equity': float(total_equity),
            'positions_open': int(len(positions)),
        })

    # --------------------------------------------------------------
    # Finalize outputs
    # --------------------------------------------------------------
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    equity_df.to_csv('reports/backtest_equity_curve.csv', index=False)
    diagnostics_df.to_csv('reports/backtest_daily_diagnostics.csv', index=False)
    if not trades_df.empty:
        trades_df.to_csv('reports/backtest_trades.csv', index=False)
    else:
        pd.DataFrame(columns=['symbol']).to_csv('reports/backtest_trades.csv', index=False)

    if equity_df.empty:
        raise ValueError('No equity curve generated.')

    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.sort_values('date').set_index('date')
    daily_returns = equity_df['equity'].pct_change().dropna()

    total_return = float(equity_df['equity'].iloc[-1] / INITIAL_CASH - 1.0)
    cagr = calculate_cagr(equity_df['equity'])
    sharpe = calculate_sharpe(daily_returns)
    sortino = calculate_sortino(daily_returns)
    max_drawdown = calculate_max_drawdown(equity_df['equity'])
    calmar = calculate_calmar(cagr, max_drawdown)

    trade_summary = summarize_trades(trades_df)

    exit_stats = {}
    if not trades_df.empty:
        grouped = trades_df.groupby('exit_reason').agg(
            trades=('pnl', 'count'),
            win_rate=('pnl', lambda s: float((s > 0).mean())),
            mean_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
            mean_return_pct=('return_pct', 'mean'),
            mean_holding_days=('holding_days', 'mean'),
        ).round(6)
        grouped.to_csv('reports/backtest_exit_reason_stats.csv')
        exit_stats = grouped.to_dict(orient='index')
    else:
        pd.DataFrame().to_csv('reports/backtest_exit_reason_stats.csv')

    summary = {
        'version': 'v4.0',
        'cutoff_date': str(cutoff_date),
        'initial_cash': INITIAL_CASH,
        'ending_equity': float(equity_df['equity'].iloc[-1]),
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'mean_positions_open': float(equity_df['positions_open'].mean()),
        'mean_exposure_estimate': float(equity_df['marked_value'].mean() / equity_df['equity'].mean()) if equity_df['equity'].mean() != 0 else 0.0,
        'trade_summary': trade_summary,
        'exit_reason_stats': exit_stats,
        'config': {
            'min_confidence': MIN_CONFIDENCE,
            'soft_min_prob': SOFT_MIN_PROB,
            'exit_threshold': EXIT_THRESHOLD,
            'top_n_per_day': TOP_N_PER_DAY,
            'max_positions': MAX_POSITIONS,
            'max_holding_days': MAX_HOLDING_DAYS,
            'stop_loss_pct': STOP_LOSS_PCT,
            'trailing_stop_pct': TRAILING_STOP_PCT,
            'max_portfolio_exposure': MAX_PORTFOLIO_EXPOSURE,
        }
    }
    save_json(summary, 'reports/backtest_summary.json')
    make_plots(equity_df, trades_df)

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------
    print('\n' + '=' * 80)
    print('Backtest Summary')
    print('=' * 80)
    print(f'Initial Cash        : {INITIAL_CASH:,.2f}')
    print(f'Ending Equity       : {equity_df["equity"].iloc[-1]:,.2f}')
    print(f'Total Return        : {total_return * 100:.2f}%')
    print(f'CAGR                : {cagr * 100:.2f}%')
    print(f'Sharpe              : {sharpe:.3f}')
    print(f'Sortino             : {sortino:.3f}')
    print(f'Max Drawdown        : {max_drawdown * 100:.2f}%')
    print(f'Calmar              : {calmar:.3f}')
    print(f'Open Positions Mean : {equity_df["positions_open"].mean():.2f}')

    print('-' * 80)
    print(f'Number of Trades    : {trade_summary["num_trades"]}')
    print(f'Win Rate            : {trade_summary["win_rate"] * 100:.2f}%')
    print(f'Avg Trade Return    : {trade_summary["avg_return_per_trade"]:.2f}%')
    print(f'Avg Holding Days    : {trade_summary["avg_holding_days"]:.2f}')
    print(f'Profit Factor       : {trade_summary["profit_factor"]:.3f}')
    print(f'Expectancy (PnL)    : {trade_summary["expectancy"]:.2f}')
    print('-' * 80)
    print('Artifacts written to reports/ and plots/')
    print('=' * 80)

    logger.info('Backtest completed successfully.')
    logger.info('=' * 80)


# ======================================================================
# 10) Entrypoint
# ======================================================================
if __name__ == '__main__':
    main()
