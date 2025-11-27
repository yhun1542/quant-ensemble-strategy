#!/usr/bin/env python3
"""
리스크 오버레이 모듈
Vol 타겟팅 + DD 기반 익스포저 레이어
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.regime import RegimeConfig, compute_spx_regime


@dataclass
class VolTargetConfig:
    """변동성 타겟팅 설정"""
    window_days: int = 63       # 3개월 정도 (252 * 0.25)
    target_vol: float = 0.15    # 연 15%
    min_leverage: float = 0.5
    max_leverage: float = 1.5


@dataclass
class DrawdownConfig:
    """Drawdown 기반 방어 설정"""
    warn_lvl: float = -0.05     # -5% 이하 경고
    cut_lvl: float = -0.10      # -10% 이하 방어 모드
    exposure_warn: float = 0.5  # 경고구간 노출
    exposure_cut: float = 0.25  # 방어구간 노출


@dataclass
class RegimeExposureConfig:
    """레짐별 익스포저 설정"""
    bull: float = 1.0
    sideways: float = 0.5
    bear: float = 0.0  # 완전 캐시 or 0.25 등으로 조정 가능


def compute_realized_vol(ret_daily: pd.Series, cfg: VolTargetConfig) -> pd.Series:
    """
    일간 수익률로부터 롤링 실현 연 변동성 추정.
    
    Parameters
    ----------
    ret_daily : pd.Series
        일간 수익률
    cfg : VolTargetConfig
        변동성 타겟팅 설정
    
    Returns
    -------
    pd.Series
        롤링 실현 연 변동성
    """
    vol_d = ret_daily.rolling(cfg.window_days).std(ddof=0)
    vol_ann = vol_d * np.sqrt(252.0)
    return vol_ann


def compute_leverage_from_vol(
    realized_vol: pd.Series,
    cfg: VolTargetConfig,
) -> pd.Series:
    """
    Vol 타깃팅 레버리지 시리즈.
    
    Parameters
    ----------
    realized_vol : pd.Series
        실현 변동성
    cfg : VolTargetConfig
        변동성 타겟팅 설정
    
    Returns
    -------
    pd.Series
        레버리지 시리즈
    """
    lev = cfg.target_vol / realized_vol
    lev = lev.clip(cfg.min_leverage, cfg.max_leverage)
    # 초기 NaN 구간은 1.0으로 처리
    lev = lev.fillna(1.0)
    return lev


def compute_drawdown_exposure(
    ret_daily: pd.Series,
    cfg: DrawdownConfig,
) -> pd.Series:
    """
    전략 자체 누적 수익률 기반 Drawdown을 계산하고,
    DD 수준에 따라 노출 스칼라(exposure)를 계산.
    
    Parameters
    ----------
    ret_daily : pd.Series
        일간 수익률
    cfg : DrawdownConfig
        Drawdown 설정
    
    Returns
    -------
    pd.Series
        노출 스칼라 시리즈
    """
    wealth = (1.0 + ret_daily).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0  # 음수 (최대낙폭)

    exposure = pd.Series(index=ret_daily.index, dtype=float)
    exposure.loc[:] = 1.0

    # cut zone
    cut_mask = dd <= cfg.cut_lvl
    warn_mask = (dd <= cfg.warn_lvl) & (dd > cfg.cut_lvl)

    exposure[warn_mask] = cfg.exposure_warn
    exposure[cut_mask] = cfg.exposure_cut

    # 초기 구간 NaN → 1.0
    exposure = exposure.fillna(1.0)
    return exposure


def compute_regime_exposure(
    spx_close: pd.Series,
    regime_cfg: RegimeConfig | None = None,
    exp_cfg: RegimeExposureConfig | None = None,
) -> pd.Series:
    """
    레짐(bull/bear/sideways)에 따른 노출 스칼라 시리즈.
    
    Parameters
    ----------
    spx_close : pd.Series
        S&P 500 종가
    regime_cfg : RegimeConfig
        레짐 판단 설정
    exp_cfg : RegimeExposureConfig
        레짐별 익스포저 설정
    
    Returns
    -------
    pd.Series
        노출 스칼라 시리즈
    """
    regime_cfg = regime_cfg or RegimeConfig()
    exp_cfg = exp_cfg or RegimeExposureConfig()

    regime = compute_spx_regime(spx_close, regime_cfg)

    exp = pd.Series(index=regime.index, dtype=float)
    exp.loc[regime == "bull"] = exp_cfg.bull
    exp.loc[regime == "sideways"] = exp_cfg.sideways
    exp.loc[regime == "bear"] = exp_cfg.bear

    # 혹시 모를 NaN → 1.0
    exp = exp.fillna(1.0)
    return exp


def apply_risk_overlays(
    ret_raw: pd.Series,
    spx_close: pd.Series,
    vol_cfg: VolTargetConfig | None = None,
    dd_cfg: DrawdownConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    regime_exp_cfg: RegimeExposureConfig | None = None,
) -> dict:
    """
    raw 엔진 수익률 위에
    - 레짐 익스포저
    - Vol 타깃팅
    - Drawdown 익스포저
    를 곱해 최종 수익률을 생성.

    룩어헤드를 피하려면:
    - 레짐, Vol, DD 계산은 '어제까지' 정보로 하고
    - 오늘 수익률에는 어제 노출(exposure.shift(1))을 곱하는 게 원칙.
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 엔진 수익률
    spx_close : pd.Series
        S&P 500 종가
    vol_cfg : VolTargetConfig
        변동성 타겟팅 설정
    dd_cfg : DrawdownConfig
        Drawdown 설정
    regime_cfg : RegimeConfig
        레짐 판단 설정
    regime_exp_cfg : RegimeExposureConfig
        레짐별 익스포저 설정
    
    Returns
    -------
    dict
        - ret_raw: 원본 수익률
        - exposure_regime: 레짐 익스포저
        - leverage_vol: Vol 타겟팅 레버리지
        - exposure_dd: DD 익스포저
        - total_exposure: 전체 익스포저
        - ret_final: 최종 수익률
    """

    vol_cfg = vol_cfg or VolTargetConfig()
    dd_cfg = dd_cfg or DrawdownConfig()
    regime_cfg = regime_cfg or RegimeConfig()
    regime_exp_cfg = regime_exp_cfg or RegimeExposureConfig()

    ret_raw = ret_raw.sort_index()
    spx_close = spx_close.sort_index().reindex(ret_raw.index, method="ffill")

    # 1) 레짐 익스포저
    exp_regime = compute_regime_exposure(spx_close, regime_cfg, regime_exp_cfg)

    # 2) Vol 타깃팅 레버리지
    realized_vol = compute_realized_vol(ret_raw, vol_cfg)
    lev_vol = compute_leverage_from_vol(realized_vol, vol_cfg)

    # 3) Drawdown 익스포저
    exp_dd = compute_drawdown_exposure(ret_raw, dd_cfg)

    # 4) 전체 익스포저 (전부 과거 기준이 돼야 하므로 shift(1))
    total_exposure = (exp_regime * lev_vol * exp_dd).shift(1).fillna(1.0)

    # 5) final 수익률
    ret_final = ret_raw * total_exposure
    ret_final.name = "ret_ensemble_with_risk"

    return {
        "ret_raw": ret_raw,
        "exposure_regime": exp_regime,
        "leverage_vol": lev_vol,
        "exposure_dd": exp_dd,
        "total_exposure": total_exposure,
        "ret_final": ret_final,
    }


if __name__ == "__main__":
    print("리스크 오버레이 모듈")
    print("="*100)
    print("Vol 타겟팅 + DD 기반 익스포저 레이어")
    print("\n사용 예시:")
    print("  from utils.risk_overlay import apply_risk_overlays")
    print("  result = apply_risk_overlays(ret_raw, spx_close)")
    print("  ret_final = result['ret_final']")
