# 01_india_data_loader.py
"""
Step 1: Download price data for 7 Indian large-cap stocks (NSE tickers),
compute daily returns, annualized mean returns r and covariance Sigma,
and save them to 'indian_data_stats.npz'.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# Chosen NIFTY large-cap tickers (Yahoo/NSE format)
TICKERS = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS','LT.NS','SBIN.NS']

# Data period
START = '2018-01-01'
END = datetime.today().strftime('%Y-%m-%d')

SAVE_PATH = 'indian_data_stats.npz'

def download_and_process(tickers=TICKERS, start=START, end=END, trading_days=252):

    # FIX: force auto_adjust=False so "Adj Close" column is available
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    if 'Adj Close' not in raw:
        raise RuntimeError("Yahoo Finance did not return 'Adj Close'. Try again or set auto_adjust=False.")

    df = raw['Adj Close'].dropna(how='all')

    # Check missing tickers
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise RuntimeError(f"Missing tickers in download: {missing}")

    # Compute daily returns
    returns = df.pct_change().dropna(how='all')

    # Mean & covariance (daily)
    mean_daily = returns.mean(axis=0).values
    cov_daily = returns.cov().values

    # Annualize
    r = mean_daily * trading_days
    Sigma = cov_daily * trading_days

    # Save
    np.savez(SAVE_PATH, r=r, Sigma=Sigma, tickers=np.array(tickers), dates=np.array(returns.index.astype(str)))
    print(f"Saved r and Sigma to {SAVE_PATH}")

    return r, Sigma, df, returns


if __name__ == '__main__':
    r, Sigma, prices, returns = download_and_process()
    print("Annualized returns r:", np.round(r, 4))
    print("Covariance matrix shape:", Sigma.shape)
