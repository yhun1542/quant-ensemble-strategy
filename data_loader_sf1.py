"""
SF1 Fundamental Data Loader with Point-in-Time Handling
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import nasdaqdatalink
import pandas as pd


@dataclass
class SF1Config:
    """Configuration for SF1 data loading"""
    api_key: str
    dimension: str = "ART"  # As-Reported, Trailing Twelve Months
    min_date: str = "2010-01-01"


def load_sf1_raw(
    tickers: List[str],
    cfg: SF1Config,
    indicators: List[str],
) -> pd.DataFrame:
    """
    Load raw SF1 fundamental data from Nasdaq Data Link.
    
    Args:
        tickers: List of ticker symbols
        cfg: SF1Config with API key and parameters
        indicators: List of indicator names (e.g., ['pe', 'pb', 'roe'])
    
    Returns:
        DataFrame with columns: ticker, dimension, calendardate, datekey, + indicators
    """
    nasdaqdatalink.ApiConfig.api_key = cfg.api_key
    
    # Build column list
    base_columns = ["ticker", "dimension", "calendardate", "datekey"]
    qopts = {
        "columns": base_columns + indicators
    }
    
    print(f"Loading SF1 data for {len(tickers)} tickers, dimension={cfg.dimension}")
    print(f"Indicators: {indicators}")
    
    try:
        df = nasdaqdatalink.get_table(
            "SHARADAR/SF1",
            ticker=tickers,
            dimension=cfg.dimension,
            calendardate={"gte": cfg.min_date},
            qopts=qopts,
            paginate=True,
        )
    except Exception as e:
        raise ValueError(f"SF1 API error: {e}")
    
    if df.empty:
        raise ValueError("SF1 API returned empty DataFrame. Check tickers/dimension/min_date.")
    
    # Convert dates
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    df["datekey"] = pd.to_datetime(df["datekey"])
    
    # Sort by ticker and datekey
    df = df.sort_values(["ticker", "datekey", "calendardate"])
    
    print(f"Loaded {len(df)} rows, date range: {df['datekey'].min()} to {df['datekey'].max()}")
    
    return df


def expand_sf1_to_daily(
    sf1_raw: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    shift_one_day: bool = True,
) -> pd.DataFrame:
    """
    Expand filing-level SF1 data to daily trading dates with point-in-time handling.
    
    This function ensures NO LOOK-AHEAD BIAS by:
    1. Using datekey (filing date) as the information availability date
    2. Optionally shifting by 1 day (shift_one_day=True) to assume market digests
       information on the day after filing
    3. Forward-filling values until the next filing
    
    Args:
        sf1_raw: Raw SF1 DataFrame from load_sf1_raw
        trading_dates: DatetimeIndex of trading days (from price data)
        shift_one_day: If True, information becomes available on day after datekey
    
    Returns:
        MultiIndex (date, ticker) DataFrame with daily fundamental values
    """
    records = []
    
    for tkr, g in sf1_raw.groupby("ticker"):
        g_sorted = g.sort_values("datekey")
        
        # Set datekey as index
        g_sorted = g_sorted.set_index("datekey")
        
        # Drop non-indicator columns
        indicator_cols = [c for c in g_sorted.columns 
                         if c not in ["ticker", "dimension", "calendardate"]]
        g_sorted = g_sorted[indicator_cols]
        
        # Reindex to trading dates with forward fill
        g_tmp = g_sorted.reindex(trading_dates, method="ffill")
        
        # Shift by 1 day to prevent look-ahead bias
        if shift_one_day:
            g_tmp = g_tmp.shift(1)
        
        g_tmp["ticker"] = tkr
        records.append(g_tmp)
    
    if not records:
        raise ValueError("No data after expansion")
    
    df_all = pd.concat(records)
    df_all = df_all.reset_index()
    df_all = df_all.rename(columns={"index": "date"})
    df_all = df_all.set_index(["date", "ticker"])
    
    print(f"Expanded to {len(df_all)} daily observations")
    print(f"Date range: {df_all.index.get_level_values('date').min()} to {df_all.index.get_level_values('date').max()}")
    
    return df_all
