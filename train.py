#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Daily Strategy Training System - Institutional Enhanced Edition v4.0
==========================================================================
Core goals:
1. Keep the original daily long-only XGBoost workflow that matches backtest.py.
2. Improve robustness, diagnostics, leakage control, and artifact outputs.
3. Preserve a large, readable, production-style single-file pipeline.
4. Keep the script structure rich enough for practical iterative research.

Main improvements versus the previous training script:
- Stronger data validation and logging.
- Explicit feature synchronization contract with backtest.py.
- Richer feature set: trend, volatility, volume, momentum, ATR, regime, lags.
- Better target diagnostics and yearly drift inspection.
- Correlation pruning + constant-feature pruning.
- Time-series CV hyperparameter search.
- Ensemble training with multiple seeds and early stopping.
- Validation threshold study to inform backtest confidence cutoffs.
- Saved metadata, feature importance, training summary, and model artifacts.
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
import warnings
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

# ======================================================================
# 2) Logging setup
# ======================================================================
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('plots', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print('=' * 80)
print('XGBoost Daily Strategy Training System - Institutional Enhanced Edition v4.0')
print('=' * 80)

# ======================================================================
# 3) Global configuration
# ======================================================================
DATA_PATH = 'daily_data.csv'
TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.15
MIN_SAMPLES_PER_SYMBOL = 80
N_DAYS_HOLD = 5
PROFIT_THRESHOLD = 0.025
ATR_TARGET_MULTIPLIER = 0.80
MIN_EXPECTED_SYMBOLS = 5
MIN_EXPECTED_ROWS = 400
CORR_THRESHOLD = 0.95
CONST_THRESHOLD = 1e-12
EARLY_STOPPING_ROUNDS = 60
N_RANDOM_SEARCH_ITER = 24
CV_SPLITS = 3
ENSEMBLE_SIZE = 5
BASE_SEED = 42

THRESHOLD_GRID = [
    0.45, 0.47, 0.49, 0.50, 0.52,
    0.54, 0.55, 0.57, 0.60, 0.62,
    0.65, 0.67, 0.70
]

NUMERIC_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
REQUIRED_COLS = ['date', 'symbol'] + NUMERIC_COLS

# ======================================================================
# 4) Utility helpers
# ======================================================================
def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def winsorize_series(s, lower=0.01, upper=0.99):
    if s.dropna().empty:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def annualize_return(total_return, periods):
    if periods <= 0:
        return 0.0
    try:
        return (1.0 + total_return) ** (252.0 / periods) - 1.0
    except Exception:
        return 0.0


def ensure_required_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def ensure_numeric_columns(df):
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def describe_dataset(df):
    summary = {
        'rows': int(len(df)),
        'symbols': int(df['symbol'].nunique()),
        'start_date': str(df['date'].min()),
        'end_date': str(df['date'].max()),
        'null_close': int(df['Close'].isna().sum()),
        'duplicate_symbol_date': int(df.duplicated(subset=['symbol', 'date']).sum()),
    }
    return summary


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ======================================================================
# 5) Data loading and validation
# ======================================================================
def load_daily_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot find {path}. Please generate it first.')

    logger.info('Loading daily data...')
    df = pd.read_csv(path, parse_dates=['date'])
    ensure_required_columns(df)
    df = ensure_numeric_columns(df)

    df = df.drop_duplicates(subset=['symbol', 'date']).copy()
    df = df.dropna(subset=['date', 'symbol', 'Close']).copy()
    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    if len(df) < MIN_EXPECTED_ROWS:
        raise ValueError(f'Dataset too small: {len(df)} rows')
    if df['symbol'].nunique() < MIN_EXPECTED_SYMBOLS:
        raise ValueError(f'Not enough symbols: {df["symbol"].nunique()}')

    desc = describe_dataset(df)
    logger.info(f'Dataset summary: {desc}')
    save_json(desc, 'reports/dataset_summary.json')
    return df


def create_time_split(df, train_split=TRAIN_SPLIT):
    all_dates = sorted(df['date'].dropna().unique())
    cutoff_idx = int(len(all_dates) * train_split)
    cutoff_idx = min(max(cutoff_idx, 1), len(all_dates) - 1)
    cutoff_date = pd.to_datetime(all_dates[cutoff_idx])

    with open('cutoff_date.txt', 'w', encoding='utf-8') as f:
        f.write(str(cutoff_date))

    train_df = df[df['date'] <= cutoff_date].copy()
    test_df = df[df['date'] > cutoff_date].copy()

    logger.info('Time split summary:')
    logger.info(f'  Train cutoff: <= {cutoff_date}')
    logger.info(f'  Train rows  : {len(train_df)}')
    logger.info(f'  Future rows : {len(test_df)}')

    split_summary = {
        'cutoff_date': str(cutoff_date),
        'train_rows': int(len(train_df)),
        'future_rows': int(len(test_df)),
        'train_symbols': int(train_df['symbol'].nunique()),
        'future_symbols': int(test_df['symbol'].nunique()),
    }
    save_json(split_summary, 'reports/time_split_summary.json')
    return cutoff_date, train_df, test_df


# ======================================================================
# 6) Feature engineering contract
# IMPORTANT: create_daily_features() must match backtest.py logic.
# ======================================================================
def create_daily_features(df):
    """
    Build a rich daily feature matrix.
    The returned feature columns are the contract used by backtest.py.
    """
    df = df.copy().sort_values(['symbol', 'date']).reset_index(drop=True)
    symbols = df['symbol'].unique()
    feature_cols = []

    # --------------------------------------------------------------
    # Initialize columns for consistency and to avoid accidental drift
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Symbol-by-symbol calculations to avoid cross-symbol leakage
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Calendar features and lags
    # --------------------------------------------------------------
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
# 7) Target construction and diagnostics
# ======================================================================
def build_training_target(df):
    df = df.copy()
    future_close = df.groupby('symbol')['Close'].shift(-N_DAYS_HOLD)
    future_return = safe_div(future_close - df['Close'], df['Close'])

    # Dynamic hurdle = max(static threshold, ATR-normalized hurdle)
    atr_hurdle = df['atr_pct'] * ATR_TARGET_MULTIPLIER
    target_hurdle = np.maximum(PROFIT_THRESHOLD, atr_hurdle)

    df['future_return'] = future_return
    df['target_hurdle'] = target_hurdle
    df['target'] = (future_return > target_hurdle).astype(int)
    return df


def save_target_diagnostics(df):
    diagnostics = {}
    diagnostics['rows_after_target'] = int(len(df))
    diagnostics['target_mean'] = float(df['target'].mean()) if len(df) else 0.0

    by_year = df.groupby(df['date'].dt.year)['target'].mean().round(6)
    by_symbol = df.groupby('symbol')['target'].mean().sort_values(ascending=False).round(6)

    diagnostics['yearly_target_mean'] = {str(k): float(v) for k, v in by_year.items()}
    diagnostics['symbol_target_mean_top10'] = {str(k): float(v) for k, v in by_symbol.head(10).items()}

    save_json(diagnostics, 'reports/target_diagnostics.json')
    by_year.to_csv('reports/target_rate_by_year.csv', header=['target_rate'])
    by_symbol.to_csv('reports/target_rate_by_symbol.csv', header=['target_rate'])

    plt.figure(figsize=(10, 4))
    by_year.plot(kind='bar', color='steelblue')
    plt.title('Target Positive Rate by Year')
    plt.ylabel('Positive Rate')
    plt.tight_layout()
    plt.savefig('plots/target_rate_by_year.png', dpi=160)
    plt.close()


# ======================================================================
# 8) Feature quality filters
# ======================================================================
def drop_constant_features(df, feature_cols, threshold=CONST_THRESHOLD):
    keep = []
    dropped = []
    for c in feature_cols:
        series = df[c]
        if series.dropna().empty:
            dropped.append(c)
            continue
        if float(series.std(skipna=True)) <= threshold:
            dropped.append(c)
            continue
        keep.append(c)
    return keep, dropped


def drop_highly_correlated_features(df, feature_cols, corr_threshold=CORR_THRESHOLD):
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    keep = [c for c in feature_cols if c not in to_drop]
    return keep, to_drop


def save_feature_quality_reports(all_features, constant_dropped, corr_dropped, final_features):
    summary = {
        'initial_feature_count': len(all_features),
        'constant_dropped_count': len(constant_dropped),
        'corr_dropped_count': len(corr_dropped),
        'final_feature_count': len(final_features),
    }
    save_json(summary, 'reports/feature_selection_summary.json')

    pd.DataFrame({'feature': all_features}).to_csv('reports/all_features.csv', index=False)
    pd.DataFrame({'feature': constant_dropped}).to_csv('reports/dropped_constant_features.csv', index=False)
    pd.DataFrame({'feature': corr_dropped}).to_csv('reports/dropped_correlated_features.csv', index=False)
    pd.DataFrame({'feature': final_features}).to_csv('reports/final_features.csv', index=False)


# ======================================================================
# 9) Validation metrics and threshold study
# ======================================================================
def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'threshold': float(threshold),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'positive_predictions': int(y_pred.sum()),
    }


def run_threshold_study(y_true, y_prob, thresholds):
    rows = [evaluate_predictions(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(rows)
    df.to_csv('reports/validation_threshold_study.csv', index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df['threshold'], df['precision'], label='Precision', marker='o')
    plt.plot(df['threshold'], df['recall'], label='Recall', marker='o')
    plt.plot(df['threshold'], df['f1'], label='F1', marker='o')
    plt.plot(df['threshold'], df['accuracy'], label='Accuracy', marker='o')
    plt.title('Validation Threshold Study')
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/validation_threshold_study.png', dpi=160)
    plt.close()
    return df


# ======================================================================
# 10) Hyperparameter search
# ======================================================================
def build_search_space(scale_pos_weight):
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=-1,
        random_state=BASE_SEED,
    )

    param_dist = {
        'n_estimators': [250, 350, 500, 700, 900],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'subsample': [0.70, 0.80, 0.90, 1.00],
        'colsample_bytree': [0.70, 0.80, 0.90, 1.00],
        'min_child_weight': [1, 2, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.4],
        'reg_alpha': [0.0, 0.05, 0.1, 0.5],
        'reg_lambda': [1.0, 2.0, 3.0, 5.0, 8.0],
    }
    return base_model, param_dist


def perform_random_search(X_train, y_train, scale_pos_weight):
    logger.info('Running hyperparameter random search...')
    base_model, param_dist = build_search_space(scale_pos_weight)
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_RANDOM_SEARCH_ITER,
        scoring='roc_auc',
        cv=tscv,
        random_state=BASE_SEED,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
    cv_results.to_csv('reports/random_search_results.csv', index=False)

    logger.info(f'Best params: {search.best_params_}')
    logger.info(f'Best CV AUC: {search.best_score_:.6f}')
    return search.best_params_, float(search.best_score_)


# ======================================================================
# 11) Ensemble training
# ======================================================================
def compute_scale_pos_weight(y):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / max(pos, 1))


def train_ensemble(X_train, y_train, X_valid, y_valid, feature_cols, best_params):
    logger.info(f'Training ensemble models: {ENSEMBLE_SIZE}')
    scale_pos_weight = compute_scale_pos_weight(y_train)
    rows = []
    ensemble_models = []

    for i in range(ENSEMBLE_SIZE):
        seed = BASE_SEED + i
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=-1,
            random_state=seed,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **best_params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        train_prob = model.predict_proba(X_train)[:, 1]
        valid_prob = model.predict_proba(X_valid)[:, 1]
        train_auc = roc_auc_score(y_train, train_prob) if len(np.unique(y_train)) > 1 else 0.5
        valid_auc = roc_auc_score(y_valid, valid_prob) if len(np.unique(y_valid)) > 1 else 0.5

        row = {
            'model_no': i + 1,
            'seed': seed,
            'best_iteration': int(model.best_iteration) if model.best_iteration is not None else -1,
            'train_auc': float(train_auc),
            'valid_auc': float(valid_auc),
        }
        rows.append(row)

        model_data = {
            'model': model,
            'features': feature_cols,
            'params': best_params,
            'seed': seed,
            'best_iteration': row['best_iteration'],
            'train_auc': row['train_auc'],
            'valid_auc': row['valid_auc'],
            'version': 'v4.0',
        }
        out_path = f'models/xgboost_daily_model_{i+1}.pkl'
        joblib.dump(model_data, out_path)
        ensemble_models.append(model_data)
        logger.info(f'  Saved {out_path} | valid_auc={valid_auc:.6f}')

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv('reports/ensemble_training_summary.csv', index=False)
    return ensemble_models, summary_df


# ======================================================================
# 12) Feature importance reports
# ======================================================================
def save_feature_importance_report(ensemble_models, feature_cols):
    importance_map = {f: [] for f in feature_cols}
    for md in ensemble_models:
        booster = md['model'].get_booster()
        score = booster.get_score(importance_type='gain')
        for f in feature_cols:
            importance_map[f].append(float(score.get(f, 0.0)))

    rows = []
    for f in feature_cols:
        vals = importance_map[f]
        rows.append({
            'feature': f,
            'mean_gain': float(np.mean(vals)),
            'median_gain': float(np.median(vals)),
            'max_gain': float(np.max(vals)),
        })

    imp_df = pd.DataFrame(rows).sort_values('mean_gain', ascending=False)
    imp_df.to_csv('reports/feature_importance_gain.csv', index=False)

    plt.figure(figsize=(10, 8))
    top = imp_df.head(25).iloc[::-1]
    plt.barh(top['feature'], top['mean_gain'], color='darkcyan')
    plt.title('Top 25 Mean Feature Gain Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance_top25.png', dpi=170)
    plt.close()
    return imp_df


# ======================================================================
# 13) Main training pipeline
# ======================================================================
def main():
    logger.info('=' * 80)
    logger.info('Starting training pipeline...')

    # --------------------------------------------------------------
    # Load and split data
    # --------------------------------------------------------------
    daily_df = load_daily_data(DATA_PATH)
    cutoff_date, train_df, future_df = create_time_split(daily_df)

    # --------------------------------------------------------------
    # Create features on training window only
    # --------------------------------------------------------------
    logger.info('Creating daily features...')
    df_feat, feature_cols = create_daily_features(train_df)
    logger.info(f'Feature matrix created with {len(feature_cols)} raw features.')

    # --------------------------------------------------------------
    # Target construction
    # --------------------------------------------------------------
    logger.info(f'Building target: hold {N_DAYS_HOLD} days, threshold {PROFIT_THRESHOLD:.2%}, ATR multiplier {ATR_TARGET_MULTIPLIER:.2f}')
    df_feat = build_training_target(df_feat)
    df_feat = df_feat.dropna(subset=['target'] + feature_cols).copy()
    logger.info(f'Rows after target + feature NaN drop: {len(df_feat)}')
    logger.info(f'Positive target rate: {df_feat["target"].mean():.4f}')
    save_target_diagnostics(df_feat)

    # --------------------------------------------------------------
    # Feature quality filtering
    # --------------------------------------------------------------
    logger.info('Running feature quality filters...')
    initial_features = feature_cols.copy()
    feature_cols, constant_dropped = drop_constant_features(df_feat, feature_cols)
    logger.info(f'Constant/empty features removed: {len(constant_dropped)}')

    feature_cols, corr_dropped = drop_highly_correlated_features(df_feat, feature_cols)
    logger.info(f'Highly correlated features removed: {len(corr_dropped)}')
    logger.info(f'Final feature count: {len(feature_cols)}')
    save_feature_quality_reports(initial_features, constant_dropped, corr_dropped, feature_cols)

    # --------------------------------------------------------------
    # Final modeling dataset
    # --------------------------------------------------------------
    df_feat = df_feat.dropna(subset=feature_cols + ['target']).copy()
    X = df_feat[feature_cols].copy()
    y = df_feat['target'].astype(int).copy()
    model_dates = df_feat['date'].copy()
    model_symbols = df_feat['symbol'].copy()

    split_idx = int(len(X) * (1.0 - VALIDATION_SPLIT))
    split_idx = min(max(split_idx, 1), len(X) - 1)
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
    d_train, d_valid = model_dates.iloc[:split_idx], model_dates.iloc[split_idx:]
    s_train, s_valid = model_symbols.iloc[:split_idx], model_symbols.iloc[split_idx:]

    logger.info(f'Train rows: {len(X_train)} | Valid rows: {len(X_valid)}')
    logger.info(f'Train positive rate: {y_train.mean():.4f} | Valid positive rate: {y_valid.mean():.4f}')

    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info(f'scale_pos_weight = {scale_pos_weight:.4f}')

    # --------------------------------------------------------------
    # Hyperparameter search
    # --------------------------------------------------------------
    best_params, best_cv_auc = perform_random_search(X_train, y_train, scale_pos_weight)

    # --------------------------------------------------------------
    # Ensemble fit
    # --------------------------------------------------------------
    ensemble_models, ensemble_summary = train_ensemble(
        X_train, y_train, X_valid, y_valid, feature_cols, best_params
    )

    # --------------------------------------------------------------
    # Ensemble validation study
    # --------------------------------------------------------------
    logger.info('Running validation ensemble study...')
    valid_probs = []
    for md in ensemble_models:
        valid_probs.append(md['model'].predict_proba(X_valid)[:, 1])
    valid_prob_mean = np.mean(np.column_stack(valid_probs), axis=1)

    valid_pred_df = pd.DataFrame({
        'date': d_valid.values,
        'symbol': s_valid.values,
        'y_true': y_valid.values,
        'y_prob': valid_prob_mean,
    })
    valid_pred_df.to_csv('reports/validation_predictions.csv', index=False)

    threshold_study = run_threshold_study(y_valid.values, valid_prob_mean, THRESHOLD_GRID)
    default_metrics = evaluate_predictions(y_valid.values, valid_prob_mean, threshold=0.55)
    save_json(default_metrics, 'reports/default_validation_metrics.json')

    # --------------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------------
    imp_df = save_feature_importance_report(ensemble_models, feature_cols)

    # --------------------------------------------------------------
    # Metadata artifact
    # --------------------------------------------------------------
    metadata = {
        'version': 'v4.0',
        'cutoff_date': str(cutoff_date),
        'data_path': DATA_PATH,
        'train_split': TRAIN_SPLIT,
        'validation_split': VALIDATION_SPLIT,
        'min_samples_per_symbol': MIN_SAMPLES_PER_SYMBOL,
        'n_days_hold': N_DAYS_HOLD,
        'profit_threshold': PROFIT_THRESHOLD,
        'atr_target_multiplier': ATR_TARGET_MULTIPLIER,
        'best_cv_auc': best_cv_auc,
        'best_params': best_params,
        'feature_cols': feature_cols,
        'n_features_final': len(feature_cols),
        'train_rows': len(X_train),
        'valid_rows': len(X_valid),
        'train_target_rate': float(y_train.mean()),
        'valid_target_rate': float(y_valid.mean()),
        'recommended_entry_threshold': 0.55,
        'recommended_exit_threshold': 0.45,
        'train_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    joblib.dump(metadata, 'models/train_metadata.pkl')
    save_json(metadata, 'reports/train_metadata.json')

    # --------------------------------------------------------------
    # Human-readable training summary
    # --------------------------------------------------------------
    best_threshold_row = threshold_study.sort_values(['f1', 'precision', 'recall'], ascending=False).iloc[0].to_dict()
    training_summary = {
        'dataset_rows_total': int(len(daily_df)),
        'dataset_symbols_total': int(daily_df['symbol'].nunique()),
        'train_rows_model': int(len(X_train)),
        'valid_rows_model': int(len(X_valid)),
        'cutoff_date': str(cutoff_date),
        'best_cv_auc': float(best_cv_auc),
        'ensemble_valid_auc_mean': float(ensemble_summary['valid_auc'].mean()),
        'ensemble_valid_auc_std': float(ensemble_summary['valid_auc'].std(ddof=0)),
        'final_feature_count': int(len(feature_cols)),
        'top5_features': imp_df.head(5)['feature'].tolist(),
        'default_validation_metrics': default_metrics,
        'best_threshold_by_f1': best_threshold_row,
    }
    save_json(training_summary, 'reports/training_summary.json')

    logger.info('Training completed successfully.')
    logger.info(f'Saved metadata -> models/train_metadata.pkl')
    logger.info(f'Saved feature list -> reports/final_features.csv')
    logger.info(f'Saved threshold study -> reports/validation_threshold_study.csv')
    logger.info(f'Saved importance plot -> plots/feature_importance_top25.png')
    logger.info('=' * 80)


# ======================================================================
# 14) Entrypoint
# ======================================================================
if __name__ == '__main__':
    main()
