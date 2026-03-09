## Improved data.py
# Improvements:
# 1. Use 'Adj Close' as 'Close' for accurate returns (adjust for dividends/splits).
# 2. Drop 'Adj Close' after renaming to avoid confusion.
# 3. Added error handling for downloads with retries.
# 4. Ensured consistent column names across intervals.
# 5. Added timezone handling for dates (make naive).
# 6. Increased symbols list with more variety.
# 7. Checked for data quality: remove duplicates, handle NaNs.
# 8. Saved data with sorted dates.
# 9. Used logging instead of print.
# 10. Added progress bar for downloads.
# 11. For 1min and 15min, ensure 'Datetime' is renamed to 'datetime' consistently.
# 12. Added validation: skip if less than expected data.
# 13. Made periods and intervals configurable.
# 14. Handled MultiIndex properly.
# 15. Added docstrings.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("🚀 Yahoo Finance Data Engine - Downloading 15min, 1min, and Daily Data")

# Expanded symbols
SYMBOLS = ["NVDA", "MU", "TSLA", "MSFT", "AMD", "INTC", "AAPL", "GOOGL", "AMZN", "META", "NFLX", "QCOM", "AVGO", "SMCI", "ARM"]

# Configurations
CONFIGS = [
    {"name": "15min", "period": "60d", "interval": "15m", "min_records": 100, "date_col": "Datetime", "rename_to": "datetime"},
    {"name": "daily", "period": "1y", "interval": "1d", "min_records": 200, "date_col": "Date", "rename_to": "date"},
    {"name": "1min", "period": "7d", "interval": "1m", "min_records": 500, "date_col": "Datetime", "rename_to": "datetime"}
]

def download_with_retry(symbol, period, interval, retries=3):
    """Download data with retries."""
    for attempt in range(retries):
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if not data.empty:
                return data
            time.sleep(1)
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
    return pd.DataFrame()

for config in CONFIGS:
    name = config["name"]
    period = config["period"]
    interval = config["interval"]
    min_records = config["min_records"]
    date_col = config["date_col"]
    rename_to = config["rename_to"]

    logging.info(f"\n📥 Downloading {interval} data (recent {period})...")
    all_data = []
    for symbol in tqdm(SYMBOLS):
        logging.info(f"   Downloading {symbol}...")
        data = download_with_retry(symbol, period, interval)
        if data.empty:
            logging.warning(f"   ❌ No data for {symbol}")
            continue

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Use Adj Close as Close
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
            data.drop('Adj Close', axis=1, inplace=True)

        df = data.reset_index()
        if date_col in df.columns:
            df.rename(columns={date_col: rename_to}, inplace=True)
        df['symbol'] = symbol

        # Make datetime naive
        if pd.api.types.is_datetime64tz_dtype(df[rename_to]):
            df[rename_to] = df[rename_to].dt.tz_localize(None)

        # Quality checks
        df.drop_duplicates(subset=[rename_to], inplace=True)
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='all', inplace=True)
        df.sort_values(rename_to, inplace=True)

        if len(df) >= min_records:
            logging.info(f"   ✅ {symbol}: {len(df)} records")
            all_data.append(df)
        else:
            logging.warning(f"   ⚠️ {symbol}: Too few records ({len(df)})")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        filename = f"{name}_data.csv" if name != "15min" else "15min_training_data.csv"
        master_df.to_csv(filename, index=False)
        logging.info(f"\n🎉 {interval} data saved: {filename}")
    else:
        logging.error(f"\n❌ {interval} data download failed")
