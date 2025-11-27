#!/usr/bin/env python3
"""
v1.2 앙상블 전략 백테스트
FV3c + ML9 앙상블에 레짐 필터 + Vol 타겟팅 + DD 방어 레이어 추가
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils.risk_overlay import (
    apply_risk_overlays,
    VolTargetConfig,
    DrawdownConfig,
    RegimeConfig,
    RegimeExposureConfig,
)


def load_engine_returns(results_dir: Path) -> tuple:
    """
    기존 엔진 결과에서 일간 수익률 로드
    
    Returns
    -------
    tuple
        (ret_fv3c, ret_ml9)
    """
    # FV3c 엔진 결과
    with open(results_dir / "factor_value_v3c_dynamic_oos.json", "r") as f:
        fv_data = json.load(f)
    
    # daily_returns가 list 형태인 경우
    if isinstance(fv_data["daily_returns"], list):
        fv_df = pd.DataFrame(fv_data["daily_returns"])
        ret_fv = pd.Series(
            fv_df["ret"].values,
            index=pd.to_datetime(fv_df["date"]),
            name="fv3c",
        ).sort_index()
    else:
        ret_fv = pd.Series(
            fv_data["daily_returns"]["values"],
            index=pd.to_datetime(fv_data["daily_returns"]["index"]),
            name="fv3c",
        ).sort_index()
    
    # ML9 엔진 결과
    with open(results_dir / "ml_xgboost_v9_ranking_oos.json", "r") as f:
        ml_data = json.load(f)
    
    # daily_returns가 list 형태인 경우
    if isinstance(ml_data["daily_returns"], list):
        ml_df = pd.DataFrame(ml_data["daily_returns"])
        ret_ml = pd.Series(
            ml_df["ret"].values,
            index=pd.to_datetime(ml_df["date"]),
            name="ml9",
        ).sort_index()
    else:
        ret_ml = pd.Series(
            ml_data["daily_returns"]["values"],
            index=pd.to_datetime(ml_data["daily_returns"]["index"]),
            name="ml9",
        ).sort_index()
    
    return ret_fv, ret_ml


def load_spx_close(data_dir: Path) -> pd.Series:
    """
    S&P 500 종가 데이터 로드
    
    Returns
    -------
    pd.Series
        S&P 500 종가
    """
    spx_df = pd.read_csv(data_dir / "spx_close.csv", index_col=0, parse_dates=True)
    return spx_df["SPX"]


def calc_monthly_metrics(ret_daily: pd.Series) -> dict:
    """
    일간 수익률로부터 월간 메트릭 계산
    
    Parameters
    ----------
    ret_daily : pd.Series
        일간 수익률
    
    Returns
    -------
    dict
        메트릭 딕셔너리
    """
    # 월간 수익률
    monthly_ret = ret_daily.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_ret) < 2:
        return {}
    
    # 연환산 수익률
    ann_return = monthly_ret.mean() * 12
    
    # 연환산 변동성
    ann_vol = monthly_ret.std() * np.sqrt(12)
    
    # Sharpe Ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    wealth = (1 + monthly_ret).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_drawdown = dd.min()
    
    # Win Rate
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret)
    
    return {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_months": len(monthly_ret),
    }


def build_ensemble_v1_0(ret_fv3c: pd.Series, ret_ml9: pd.Series, w_fv: float = 0.6, w_ml: float = 0.4) -> pd.Series:
    """
    v1.0 앙상블 (레이어 없음)
    
    Parameters
    ----------
    ret_fv3c : pd.Series
        FV3c 엔진 수익률
    ret_ml9 : pd.Series
        ML9 엔진 수익률
    w_fv : float
        FV3c 가중치
    w_ml : float
        ML9 가중치
    
    Returns
    -------
    pd.Series
        앙상블 수익률
    """
    df = pd.concat([ret_fv3c.rename("fv"), ret_ml9.rename("ml")], axis=1).dropna()
    ret_ensemble = w_fv * df["fv"] + w_ml * df["ml"]
    ret_ensemble.name = "ret_ensemble_v1_0"
    return ret_ensemble


def build_ensemble_v1_2(
    ret_fv3c: pd.Series,
    ret_ml9: pd.Series,
    spx_close: pd.Series,
    w_fv: float = 0.6,
    w_ml: float = 0.4,
    vol_cfg: VolTargetConfig = None,
    dd_cfg: DrawdownConfig = None,
    regime_cfg: RegimeConfig = None,
    regime_exp_cfg: RegimeExposureConfig = None,
) -> dict:
    """
    v1.2 앙상블 (레짐 + Vol + DD 레이어)
    
    Parameters
    ----------
    ret_fv3c : pd.Series
        FV3c 엔진 수익률
    ret_ml9 : pd.Series
        ML9 엔진 수익률
    spx_close : pd.Series
        S&P 500 종가
    w_fv : float
        FV3c 가중치
    w_ml : float
        ML9 가중치
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
        리스크 레이어 결과
    """
    # 1) v1.0 앙상블 (raw)
    ret_raw = build_ensemble_v1_0(ret_fv3c, ret_ml9, w_fv, w_ml)
    
    # 2) 리스크 레이어 적용
    risk_result = apply_risk_overlays(
        ret_raw=ret_raw,
        spx_close=spx_close,
        vol_cfg=vol_cfg or VolTargetConfig(),
        dd_cfg=dd_cfg or DrawdownConfig(),
        regime_cfg=regime_cfg or RegimeConfig(),
        regime_exp_cfg=regime_exp_cfg or RegimeExposureConfig(),
    )
    
    return risk_result


def compare_periods(ret: pd.Series, split_date: str = "2023-06-06") -> dict:
    """
    구간별 성과 비교
    
    Parameters
    ----------
    ret : pd.Series
        일간 수익률
    split_date : str
        분할 날짜
    
    Returns
    -------
    dict
        구간별 메트릭
    """
    split_ts = pd.Timestamp(split_date)
    
    # 전체
    metrics_full = calc_monthly_metrics(ret)
    
    # IS (약세장)
    ret_is = ret.loc[:split_ts]
    metrics_is = calc_monthly_metrics(ret_is)
    
    # OOS (강세장)
    ret_oos = ret.loc[split_ts:]
    metrics_oos = calc_monthly_metrics(ret_oos)
    
    return {
        "full": metrics_full,
        "bear_market": metrics_is,
        "bull_market": metrics_oos,
    }


def main():
    """메인 실행 함수"""
    print("="*100)
    print("v1.2 앙상블 전략 백테스트")
    print("="*100)
    
    # 경로 설정
    base_dir = Path(".")
    results_dir = base_dir / "results"
    data_dir = base_dir / "data"
    
    # 1) 데이터 로드
    print("\n1. 데이터 로딩...")
    ret_fv, ret_ml = load_engine_returns(results_dir)
    spx_close = load_spx_close(data_dir)
    
    print(f"   FV3c: {len(ret_fv)} 일")
    print(f"   ML9: {len(ret_ml)} 일")
    print(f"   SPX: {len(spx_close)} 일")
    
    # 2) v1.0 앙상블 (레이어 없음)
    print("\n2. v1.0 앙상블 (레이어 없음)...")
    ret_v1_0 = build_ensemble_v1_0(ret_fv, ret_ml, w_fv=0.6, w_ml=0.4)
    metrics_v1_0 = compare_periods(ret_v1_0)
    
    print("\n   v1.0 성과:")
    print(f"   전체 Sharpe: {metrics_v1_0['full']['sharpe']:.4f}")
    print(f"   약세장 Sharpe: {metrics_v1_0['bear_market']['sharpe']:.4f}")
    print(f"   강세장 Sharpe: {metrics_v1_0['bull_market']['sharpe']:.4f}")
    
    # 3) v1.2 앙상블 (레짐 + Vol + DD)
    print("\n3. v1.2 앙상블 (레짐 + Vol + DD)...")
    
    risk_result = build_ensemble_v1_2(
        ret_fv3c=ret_fv,
        ret_ml9=ret_ml,
        spx_close=spx_close,
        w_fv=0.6,
        w_ml=0.4,
        vol_cfg=VolTargetConfig(
            window_days=63,
            target_vol=0.15,
            min_leverage=0.5,
            max_leverage=1.5,
        ),
        dd_cfg=DrawdownConfig(
            warn_lvl=-0.05,
            cut_lvl=-0.10,
            exposure_warn=0.5,
            exposure_cut=0.25,
        ),
        regime_cfg=RegimeConfig(
            ma_long=200,
            ma_short=50,
            upper_band=0.01,
            lower_band=-0.01,
        ),
        regime_exp_cfg=RegimeExposureConfig(
            bull=1.0,
            sideways=0.75,
            bear=0.5,  # 공격적 설정 (최적화 결과)
        ),
    )
    
    ret_v1_2 = risk_result["ret_final"]
    metrics_v1_2 = compare_periods(ret_v1_2)
    
    print("\n   v1.2 성과:")
    print(f"   전체 Sharpe: {metrics_v1_2['full']['sharpe']:.4f}")
    print(f"   약세장 Sharpe: {metrics_v1_2['bear_market']['sharpe']:.4f}")
    print(f"   강세장 Sharpe: {metrics_v1_2['bull_market']['sharpe']:.4f}")
    
    # 4) 비교 분석
    print("\n" + "="*100)
    print("v1.0 vs v1.2 비교")
    print("="*100)
    
    comparison = pd.DataFrame({
        "v1.0": [
            metrics_v1_0["full"]["sharpe"],
            metrics_v1_0["bear_market"]["sharpe"],
            metrics_v1_0["bull_market"]["sharpe"],
            metrics_v1_0["full"]["ann_return"],
            metrics_v1_0["full"]["ann_vol"],
            metrics_v1_0["full"]["max_drawdown"],
        ],
        "v1.2": [
            metrics_v1_2["full"]["sharpe"],
            metrics_v1_2["bear_market"]["sharpe"],
            metrics_v1_2["bull_market"]["sharpe"],
            metrics_v1_2["full"]["ann_return"],
            metrics_v1_2["full"]["ann_vol"],
            metrics_v1_2["full"]["max_drawdown"],
        ],
    }, index=[
        "Sharpe (전체)",
        "Sharpe (약세장)",
        "Sharpe (강세장)",
        "연수익률",
        "연변동성",
        "Max DD",
    ])
    
    comparison["변화"] = comparison["v1.2"] - comparison["v1.0"]
    comparison["변화율 (%)"] = (comparison["v1.2"] / comparison["v1.0"] - 1) * 100
    
    print("\n" + comparison.to_string())
    
    # 5) 결과 저장
    print("\n5. 결과 저장...")
    
    output = {
        "version": "v1.2",
        "description": "FV3c + ML9 앙상블 + 레짐 필터 + Vol 타겟팅 + DD 방어",
        "config": {
            "weights": {"fv3c": 0.6, "ml9": 0.4},
            "vol_target": 0.15,
            "dd_warn": -0.05,
            "dd_cut": -0.10,
            "regime_bull": 1.0,
            "regime_sideways": 0.5,
            "regime_bear": 0.0,
        },
        "metrics": {
            "v1_0": {
                "full": metrics_v1_0["full"],
                "bear_market": metrics_v1_0["bear_market"],
                "bull_market": metrics_v1_0["bull_market"],
            },
            "v1_2": {
                "full": metrics_v1_2["full"],
                "bear_market": metrics_v1_2["bear_market"],
                "bull_market": metrics_v1_2["bull_market"],
            },
        },
        "daily_returns": {
            "v1_0": {
                "index": [d.strftime("%Y-%m-%d") for d in ret_v1_0.index],
                "values": ret_v1_0.astype(float).tolist(),
            },
            "v1_2": {
                "index": [d.strftime("%Y-%m-%d") for d in ret_v1_2.index],
                "values": ret_v1_2.astype(float).tolist(),
            },
        },
        "exposure": {
            "regime": {
                "index": [d.strftime("%Y-%m-%d") for d in risk_result["exposure_regime"].index],
                "values": risk_result["exposure_regime"].astype(float).tolist(),
            },
            "vol_leverage": {
                "index": [d.strftime("%Y-%m-%d") for d in risk_result["leverage_vol"].index],
                "values": risk_result["leverage_vol"].astype(float).tolist(),
            },
            "dd": {
                "index": [d.strftime("%Y-%m-%d") for d in risk_result["exposure_dd"].index],
                "values": risk_result["exposure_dd"].astype(float).tolist(),
            },
            "total": {
                "index": [d.strftime("%Y-%m-%d") for d in risk_result["total_exposure"].index],
                "values": risk_result["total_exposure"].astype(float).tolist(),
            },
        },
    }
    
    output_path = results_dir / "ensemble_v1_2_backtest.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"   결과 저장 완료: {output_path}")
    
    print("\n✅ v1.2 백테스트 완료!")


if __name__ == "__main__":
    main()
