#!/usr/bin/env python3
"""
Factors 계산 모듈
Signal Smoothing을 지원하는 팩터 계산 함수들
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def compute_momentum_60d(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    60일 모멘텀 팩터 계산
    
    Parameters
    ----------
    prices : pd.DataFrame
        실제 가격 (index=date, columns=tickers)
    signal_prices : pd.DataFrame, optional
        시그널용 가격 (없으면 prices 사용)
    
    Returns
    -------
    pd.DataFrame
        60일 모멘텀 (index=date, columns=tickers)
    """
    px = signal_prices if signal_prices is not None else prices
    mom_60 = px / px.shift(60) - 1.0
    return mom_60


def compute_volatility_30d(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    30일 변동성 팩터 계산
    
    Parameters
    ----------
    prices : pd.DataFrame
        실제 가격
    signal_prices : pd.DataFrame, optional
        시그널용 가격 (없으면 prices 사용)
    
    Returns
    -------
    pd.DataFrame
        30일 변동성 (index=date, columns=tickers)
    """
    px = signal_prices if signal_prices is not None else prices
    vol_30 = px.pct_change().rolling(30).std()
    return vol_30


def compute_value_proxy(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Value Proxy 팩터 계산
    
    간단한 구현: 가격의 역수 (실제로는 fundamentals 필요)
    
    Parameters
    ----------
    prices : pd.DataFrame
        실제 가격
    signal_prices : pd.DataFrame, optional
        시그널용 가격 (없으면 prices 사용)
    
    Returns
    -------
    pd.DataFrame
        Value proxy (index=date, columns=tickers)
    """
    px = signal_prices if signal_prices is not None else prices
    # 간단한 구현: 가격의 역수
    # 실제로는 fundamentals (book value, earnings 등) 필요
    value_proxy = 1.0 / px
    return value_proxy


def compute_all_factors(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    모든 팩터를 한번에 계산
    
    Parameters
    ----------
    prices : pd.DataFrame
        실제 가격
    signal_prices : pd.DataFrame, optional
        시그널용 가격 (없으면 prices 사용)
    
    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) x columns (factors)
    """
    mom_60 = compute_momentum_60d(prices, signal_prices)
    vol_30 = compute_volatility_30d(prices, signal_prices)
    value = compute_value_proxy(prices, signal_prices)
    
    # Stack to MultiIndex format
    factors_list = []
    
    factors_list.append(mom_60.stack().to_frame("momentum_60d"))
    factors_list.append(vol_30.stack().to_frame("volatility_30d"))
    factors_list.append(value.stack().to_frame("value_proxy"))
    
    factors = pd.concat(factors_list, axis=1)
    factors.index.names = ["date", "ticker"]
    
    # value_proxy_inv 추가 (ML 엔진용)
    factors["value_proxy_inv"] = 1.0 / factors["value_proxy"]
    
    return factors


if __name__ == "__main__":
    print("Factors 계산 모듈")
    print("="*100)
    print("Signal Smoothing을 지원하는 팩터 계산 함수들")
    print("\n사용 예시:")
    print("  from utils.factors import compute_all_factors")
    print("  factors = compute_all_factors(prices, signal_prices)")
