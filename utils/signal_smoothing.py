#!/usr/bin/env python3
"""
Signal Smoothing 모듈
리밸런싱 시그널을 특정 하루 종가에만 의존하지 않게 만들기
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


@dataclass
class SignalSmoothingConfig:
    """Signal Smoothing 설정"""
    window_days: int = 3  # 월초 몇 일 평균을 사용할지
    method: str = "mean"  # mean, median, vwap 등


def get_monthly_signal_dates(
    index: pd.DatetimeIndex,
    window: int = 3
) -> Dict[Tuple[int, int], List[pd.Timestamp]]:
    """
    각 달의 첫 window개 거래일 리스트를 반환.
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        거래일 인덱스
    window : int
        월초 몇 일을 사용할지
    
    Returns
    -------
    dict
        {(year, month): [date1, date2, ...]} 형태
    
    Examples
    --------
    >>> dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
    >>> signal_dates = get_monthly_signal_dates(dates, window=3)
    >>> signal_dates[(2023, 1)]
    [Timestamp('2023-01-02'), Timestamp('2023-01-03'), Timestamp('2023-01-04')]
    """
    df = pd.DataFrame(index=index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    
    groups = df.groupby(["year", "month"])
    signal_dates = {}
    
    for (y, m), g in groups:
        days = list(g.index[:window])
        if len(days) > 0:
            signal_dates[(y, m)] = days
    
    return signal_dates


def compute_signal_prices(
    prices: pd.DataFrame,
    window: int = 3,
    method: str = "mean"
) -> Dict[Tuple[int, int], pd.Series]:
    """
    월별 시그널용 '가상 가격' 시리즈 생성.
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터 (index: date, columns: tickers)
    window : int
        월초 몇 일 평균을 사용할지
    method : str
        평균 방법 (mean, median)
    
    Returns
    -------
    dict
        {(year, month): Series(ticker -> price)} 형태
    
    Examples
    --------
    >>> prices = pd.DataFrame(...)  # date x ticker
    >>> signal_prices = compute_signal_prices(prices, window=3)
    >>> signal_prices[(2023, 1)]
    ticker1    100.5
    ticker2    200.3
    ...
    """
    signal_dates = get_monthly_signal_dates(prices.index, window)
    signal_price_by_month = {}
    
    for (y, m), days in signal_dates.items():
        px_window = prices.loc[days]
        
        if method == "mean":
            sig_px = px_window.mean(axis=0)
        elif method == "median":
            sig_px = px_window.median(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        signal_price_by_month[(y, m)] = sig_px
    
    return signal_price_by_month


def create_signal_price_dataframe(
    prices: pd.DataFrame,
    window: int = 3,
    method: str = "mean"
) -> pd.DataFrame:
    """
    시그널용 가상 가격 DataFrame 생성.
    각 월의 첫 거래일에 해당 월의 시그널 가격을 할당.
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터 (index: date, columns: tickers)
    window : int
        월초 몇 일 평균을 사용할지
    method : str
        평균 방법 (mean, median)
    
    Returns
    -------
    pd.DataFrame
        시그널 가격 DataFrame (index: date, columns: tickers)
        각 월의 첫 거래일에만 값이 있고, 나머지는 NaN
        forward fill하여 사용
    
    Examples
    --------
    >>> prices = pd.DataFrame(...)  # date x ticker
    >>> signal_prices = create_signal_price_dataframe(prices, window=3)
    >>> signal_prices.ffill()  # forward fill하여 사용
    """
    signal_prices_dict = compute_signal_prices(prices, window, method)
    
    # 빈 DataFrame 생성
    signal_df = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    
    # 각 월의 첫 거래일에 시그널 가격 할당
    for (y, m), sig_px in signal_prices_dict.items():
        # 해당 월의 첫 거래일 찾기
        month_dates = prices.loc[(prices.index.year == y) & (prices.index.month == m)].index
        if len(month_dates) > 0:
            first_date = month_dates[0]
            signal_df.loc[first_date] = sig_px
    
    return signal_df


def get_rebalance_dates_with_offset(
    index: pd.DatetimeIndex,
    offset: int = 0
) -> List[pd.Timestamp]:
    """
    리밸런싱 날짜 리스트 반환 (오프셋 적용)
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        거래일 인덱스
    offset : int
        월초 몇 번째 거래일을 사용할지 (0=첫날, 1=둘째날, ...)
    
    Returns
    -------
    list
        리밸런싱 날짜 리스트
    
    Examples
    --------
    >>> dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
    >>> rebal_dates = get_rebalance_dates_with_offset(dates, offset=0)
    >>> rebal_dates[0]
    Timestamp('2023-01-02')  # 1월 첫 거래일
    """
    df = pd.DataFrame(index=index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    
    groups = df.groupby(["year", "month"])
    rebal_dates = []
    
    for (y, m), g in groups:
        if len(g) > offset:
            rebal_dates.append(g.index[offset])
    
    return rebal_dates


def apply_signal_smoothing_to_returns(
    ret_raw: pd.Series,
    prices: pd.DataFrame,
    window: int = 3,
    method: str = "mean"
) -> pd.Series:
    """
    Signal Smoothing을 적용한 수익률 계산.
    
    이 함수는 간단한 근사 방법으로,
    실제로는 각 엔진에서 signal_prices를 사용하여
    팩터/랭킹을 재계산해야 합니다.
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    prices : pd.DataFrame
        가격 데이터
    window : int
        월초 몇 일 평균을 사용할지
    method : str
        평균 방법
    
    Returns
    -------
    pd.Series
        Signal Smoothing 적용 수익률
    """
    # 이 함수는 placeholder입니다.
    # 실제로는 각 엔진의 백테스트 로직을 수정해야 합니다.
    
    # 간단한 근사: 리밸 날짜의 수익률을 window일 평균으로 대체
    ret_smoothed = ret_raw.copy()
    
    rebal_dates = get_rebalance_dates_with_offset(ret_raw.index, offset=0)
    
    for rebal_date in rebal_dates:
        # 리밸 날짜 이후 window일의 수익률 평균
        idx = ret_raw.index.get_loc(rebal_date)
        if idx + window <= len(ret_raw):
            window_ret = ret_raw.iloc[idx:idx+window].mean()
            ret_smoothed.iloc[idx] = window_ret
    
    return ret_smoothed


if __name__ == "__main__":
    print("Signal Smoothing 모듈")
    print("="*100)
    print("리밸런싱 시그널을 특정 하루 종가에만 의존하지 않게 만들기")
    print("\n사용 예시:")
    print("  from utils.signal_smoothing import compute_signal_prices, create_signal_price_dataframe")
    print("  signal_prices = compute_signal_prices(prices, window=3)")
    print("  signal_df = create_signal_price_dataframe(prices, window=3)")
    print("  signal_df_filled = signal_df.ffill()")
