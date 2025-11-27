#!/usr/bin/env python3
"""
레짐 필터 모듈
S&P 500 200일선 기반 시장 레짐 판단
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Regime = Literal["bull", "bear", "sideways"]


@dataclass
class RegimeConfig:
    """레짐 판단 설정"""
    ma_long: int = 200      # 200일 장기 이동평균
    ma_short: int = 50      # 선택적: 추세 강도 확인용
    upper_band: float = 0.01  # 1% 이상 상단 이탈 → bull
    lower_band: float = -0.01 # -1% 이하 하단 이탈 → bear


def compute_spx_regime(
    spx_close: pd.Series,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """
    S&P 500 종가 시계열을 받아 일별 레짐(bull / bear / sideways) 시리즈를 반환.

    Parameters
    ----------
    spx_close : pd.Series
        index: DatetimeIndex, values: S&P 500 종가
    config : RegimeConfig
        레짐 판단 기준 파라미터

    Returns
    -------
    pd.Series
        index: DatetimeIndex, values: {"bull", "bear", "sideways"}
    
    Examples
    --------
    >>> spx_close = load_spx_close(...)
    >>> regime_series = compute_spx_regime(spx_close)
    >>> regime_series['2022-10-01']
    'bear'
    """
    cfg = config or RegimeConfig()

    spx = spx_close.sort_index().copy()
    ma_long = spx.rolling(cfg.ma_long).mean()
    ma_short = spx.rolling(cfg.ma_short).mean()

    # 현재 가격 대비 200일선 괴리율
    diff_long = (spx - ma_long) / ma_long

    regime = pd.Series(index=spx.index, dtype="object")

    # 기본값: sideways
    regime.loc[:] = "sideways"

    # bull 조건: 200일선 위 + 50일선도 위 + 괴리율 상단
    bull_mask = (
        (diff_long >= cfg.upper_band)
        & (ma_short > ma_long)
    )
    # bear 조건: 200일선 아래 + 괴리율 하단
    bear_mask = (diff_long <= cfg.lower_band)

    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"

    return regime


if __name__ == "__main__":
    print("레짐 필터 모듈")
    print("="*100)
    print("S&P 500 200일선 기반 시장 레짐 판단")
    print("\n사용 예시:")
    print("  from utils.regime import compute_spx_regime, RegimeConfig")
    print("  regime_series = compute_spx_regime(spx_close)")
    print("  print(regime_series.value_counts())")
