#!/usr/bin/env python3
"""
Signal Prices 유틸리티
엔진 레벨 Signal Smoothing을 위한 공통 모듈
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class SignalSmoothingConfig:
    """Signal Smoothing 설정"""
    window: int = 3  # 월초 몇 거래일을 평균낼지 (v1.4 기본값: 3)


def get_monthly_windows(
    index: pd.DatetimeIndex,
    cfg: SignalSmoothingConfig | None = None,
) -> Dict[Tuple[int, int], List[pd.Timestamp]]:
    """
    각 (year, month)마다 '월초 window개 거래일' 리스트를 반환.
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        거래일 인덱스
    cfg : SignalSmoothingConfig, optional
        Signal Smoothing 설정
    
    Returns
    -------
    dict
        {(year, month): [date1, date2, date3]} 형태
    
    Examples
    --------
    >>> dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
    >>> windows = get_monthly_windows(dates, SignalSmoothingConfig(window=3))
    >>> windows[(2023, 1)]
    [Timestamp('2023-01-02'), Timestamp('2023-01-03'), Timestamp('2023-01-04')]
    """
    cfg = cfg or SignalSmoothingConfig()
    df = pd.DataFrame(index=index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    
    out: Dict[Tuple[int, int], List[pd.Timestamp]] = {}
    for (y, m), g in df.groupby(["year", "month"]):
        days = list(g.index[: cfg.window])
        if days:
            out[(y, m)] = days
    return out


def build_signal_price_df(
    prices: pd.DataFrame,
    cfg: SignalSmoothingConfig | None = None,
) -> pd.DataFrame:
    """
    Signal Smoothing용 가격 DataFrame 생성.
    
    월초 window개 거래일의 평균 가격을 계산하여,
    윈도우 마지막 날짜에 할당합니다.
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터 (index=date, columns=tickers)
    cfg : SignalSmoothingConfig, optional
        Signal Smoothing 설정
    
    Returns
    -------
    pd.DataFrame
        Signal prices (index=date, columns=tickers)
        - smoothing window 마지막 날짜에만 값이 있고,
          나머지 날짜는 NaN
        - 사용 시 .reindex(prices.index).ffill()로 확장
    
    Examples
    --------
    >>> prices = pd.DataFrame(...)  # date x ticker
    >>> cfg = SignalSmoothingConfig(window=3)
    >>> signal_df_raw = build_signal_price_df(prices, cfg)
    >>> signal_df = signal_df_raw.reindex(prices.index).ffill()
    
    Notes
    -----
    룩어헤드 방지:
    - 월초 3일 (d0, d1, d2) 평균 가격을 d2에 할당
    - 실제 리밸은 d2 종가 이후 (d3부터) 시작
    - d0~d2는 "신호 쌓는 기간", d3부터 "액션 기간"
    """
    cfg = cfg or SignalSmoothingConfig()
    prices = prices.sort_index()
    
    windows = get_monthly_windows(prices.index, cfg)
    signal_rows = {}
    
    for (y, m), days in windows.items():
        px_window = prices.loc[days]           # (window, tickers)
        sig_px = px_window.mean(axis=0)       # (tickers,) 평균
        signal_date = days[-1]                # 윈도우 마지막 날을 시그널 날짜로
        signal_rows[signal_date] = sig_px
    
    signal_df = pd.DataFrame.from_dict(signal_rows, orient="index")
    signal_df.index.name = prices.index.name
    signal_df = signal_df.sort_index()
    
    return signal_df


def get_rebalance_dates_from_signal_df(
    signal_df: pd.DataFrame,
) -> List[pd.Timestamp]:
    """
    Signal DataFrame에서 리밸런싱 날짜 추출.
    
    Parameters
    ----------
    signal_df : pd.DataFrame
        build_signal_price_df()로 생성된 signal prices
    
    Returns
    -------
    list
        리밸런싱 날짜 리스트 (signal_df의 index)
    
    Examples
    --------
    >>> signal_df_raw = build_signal_price_df(prices, cfg)
    >>> rebalance_dates = get_rebalance_dates_from_signal_df(signal_df_raw)
    """
    return sorted(signal_df.index.tolist())


def expand_signal_prices(
    signal_df_raw: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Signal prices를 전체 날짜로 확장 (forward fill).
    
    Parameters
    ----------
    signal_df_raw : pd.DataFrame
        build_signal_price_df()로 생성된 원본 signal prices
    target_index : pd.DatetimeIndex
        확장할 대상 날짜 인덱스 (보통 prices.index)
    
    Returns
    -------
    pd.DataFrame
        확장된 signal prices (index=target_index, columns=tickers)
    
    Examples
    --------
    >>> signal_df_raw = build_signal_price_df(prices, cfg)
    >>> signal_df = expand_signal_prices(signal_df_raw, prices.index)
    """
    return signal_df_raw.reindex(target_index).ffill()


if __name__ == "__main__":
    print("Signal Prices 유틸리티")
    print("="*100)
    print("엔진 레벨 Signal Smoothing을 위한 공통 모듈")
    print("\n사용 예시:")
    print("  from utils.signal_prices import SignalSmoothingConfig, build_signal_price_df")
    print("  cfg = SignalSmoothingConfig(window=3)")
    print("  signal_df_raw = build_signal_price_df(prices, cfg)")
    print("  signal_df = signal_df_raw.reindex(prices.index).ffill()")
    print("  rebalance_dates = sorted(signal_df_raw.index)")
