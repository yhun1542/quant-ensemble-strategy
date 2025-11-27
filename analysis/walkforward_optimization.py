#!/usr/bin/env python3
"""
Walk-forward 최적화 프레임워크
과적합 방지를 위한 IS/OOS 구간 분리 파라미터 최적화
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from itertools import product
sys.path.append(str(Path(__file__).parent.parent))

from utils.risk_overlay import (
    apply_risk_overlays,
    VolTargetConfig,
    DrawdownConfig,
    RegimeConfig,
    RegimeExposureConfig,
)


def load_data():
    """데이터 로드"""
    # S&P 500
    spx_df = pd.read_csv("data/spx_close.csv", index_col=0, parse_dates=True)
    spx_close = spx_df["SPX"]
    
    # FV3c 엔진 결과
    with open("results/factor_value_v3c_dynamic_oos.json", "r") as f:
        fv_data = json.load(f)
    
    fv_df = pd.DataFrame(fv_data["daily_returns"])
    ret_fv = pd.Series(
        fv_df["ret"].values,
        index=pd.to_datetime(fv_df["date"]),
        name="fv3c",
    ).sort_index()
    
    # ML9 엔진 결과
    with open("results/ml_xgboost_v9_ranking_oos.json", "r") as f:
        ml_data = json.load(f)
    
    ml_df = pd.DataFrame(ml_data["daily_returns"])
    ret_ml = pd.Series(
        ml_df["ret"].values,
        index=pd.to_datetime(ml_df["date"]),
        name="ml9",
    ).sort_index()
    
    # v1.0 앙상블 (60:40)
    df = pd.concat([ret_fv.rename("fv"), ret_ml.rename("ml")], axis=1).dropna()
    ret_v1_0 = 0.6 * df["fv"] + 0.4 * df["ml"]
    ret_v1_0.name = "ret_v1_0"
    
    return spx_close, ret_v1_0


def calc_sharpe(ret_daily: pd.Series) -> float:
    """
    일간 수익률로부터 Sharpe Ratio 계산
    
    Parameters
    ----------
    ret_daily : pd.Series
        일간 수익률
    
    Returns
    -------
    float
        Sharpe Ratio
    """
    if len(ret_daily) < 2:
        return 0.0
    
    monthly_ret = ret_daily.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_ret) < 2:
        return 0.0
    
    ann_return = monthly_ret.mean() * 12
    ann_vol = monthly_ret.std() * np.sqrt(12)
    
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    return sharpe


def test_regime_config(
    ret_raw: pd.Series,
    spx_close: pd.Series,
    bull: float,
    sideways: float,
    bear: float,
) -> dict:
    """
    특정 레짐 익스포저 설정으로 백테스트
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    spx_close : pd.Series
        S&P 500 종가
    bull : float
        Bull 레짐 익스포저
    sideways : float
        Sideways 레짐 익스포저
    bear : float
        Bear 레짐 익스포저
    
    Returns
    -------
    dict
        성과 메트릭
    """
    regime_exp_cfg = RegimeExposureConfig(
        bull=bull,
        sideways=sideways,
        bear=bear,
    )
    
    risk_result = apply_risk_overlays(
        ret_raw=ret_raw,
        spx_close=spx_close,
        vol_cfg=VolTargetConfig(window_days=63, target_vol=0.15),
        dd_cfg=DrawdownConfig(warn_lvl=-0.05, cut_lvl=-0.10),
        regime_cfg=RegimeConfig(),
        regime_exp_cfg=regime_exp_cfg,
    )
    
    ret_final = risk_result["ret_final"]
    sharpe = calc_sharpe(ret_final)
    
    # 월간 메트릭
    monthly_ret = ret_final.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_ret) >= 2:
        ann_return = monthly_ret.mean() * 12
        ann_vol = monthly_ret.std() * np.sqrt(12)
        
        wealth = (1 + monthly_ret).cumprod()
        running_max = wealth.cummax()
        dd = wealth / running_max - 1
        max_dd = dd.min()
    else:
        ann_return = 0
        ann_vol = 0
        max_dd = 0
    
    return {
        "bull": bull,
        "sideways": sideways,
        "bear": bear,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
    }


def grid_search_regime_params(
    ret_raw: pd.Series,
    spx_close: pd.Series,
    bull_range: list = [1.0],
    sideways_range: list = [0.5, 0.75, 1.0],
    bear_range: list = [0.0, 0.25, 0.5],
) -> pd.DataFrame:
    """
    레짐 익스포저 파라미터 그리드 서치
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    spx_close : pd.Series
        S&P 500 종가
    bull_range : list
        Bull 레짐 익스포저 범위
    sideways_range : list
        Sideways 레짐 익스포저 범위
    bear_range : list
        Bear 레짐 익스포저 범위
    
    Returns
    -------
    pd.DataFrame
        그리드 서치 결과
    """
    results = []
    
    total_configs = len(bull_range) * len(sideways_range) * len(bear_range)
    print(f"\n그리드 서치 시작: {total_configs}개 설정 테스트")
    
    for i, (bull, sideways, bear) in enumerate(product(bull_range, sideways_range, bear_range), 1):
        print(f"  [{i}/{total_configs}] bull={bull}, sideways={sideways}, bear={bear}", end="")
        
        result = test_regime_config(ret_raw, spx_close, bull, sideways, bear)
        results.append(result)
        
        print(f" → Sharpe={result['sharpe']:.4f}")
    
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)
    
    return df


def walkforward_optimize(
    ret_raw: pd.Series,
    spx_close: pd.Series,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
) -> dict:
    """
    Walk-forward 최적화
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    spx_close : pd.Series
        S&P 500 종가
    is_start : str
        IS 구간 시작일
    is_end : str
        IS 구간 종료일
    oos_start : str
        OOS 구간 시작일
    oos_end : str
        OOS 구간 종료일
    
    Returns
    -------
    dict
        최적화 결과
    """
    print("="*100)
    print("Walk-forward 최적화")
    print("="*100)
    
    # IS 구간 데이터
    is_ret = ret_raw.loc[is_start:is_end]
    is_spx = spx_close.loc[is_start:is_end]
    
    print(f"\nIS 구간: {is_start} ~ {is_end}")
    print(f"  거래일 수: {len(is_ret)}")
    print(f"  기간: {(pd.Timestamp(is_end) - pd.Timestamp(is_start)).days} 일")
    
    # OOS 구간 데이터
    oos_ret = ret_raw.loc[oos_start:oos_end]
    oos_spx = spx_close.loc[oos_start:oos_end]
    
    print(f"\nOOS 구간: {oos_start} ~ {oos_end}")
    print(f"  거래일 수: {len(oos_ret)}")
    print(f"  기간: {(pd.Timestamp(oos_end) - pd.Timestamp(oos_start)).days} 일")
    
    # 1) IS 구간에서 파라미터 최적화
    print("\n" + "-"*100)
    print("1. IS 구간에서 파라미터 최적화")
    print("-"*100)
    
    is_results = grid_search_regime_params(
        ret_raw=is_ret,
        spx_close=is_spx,
        bull_range=[1.0],
        sideways_range=[0.5, 0.75, 1.0],
        bear_range=[0.0, 0.25, 0.5],
    )
    
    print("\nIS 구간 상위 5개 설정:")
    print(is_results.head(5).to_string(index=False))
    
    # 최적 파라미터
    best_params = is_results.iloc[0]
    print(f"\n최적 파라미터 (IS 구간 기준):")
    print(f"  bull={best_params['bull']}, sideways={best_params['sideways']}, bear={best_params['bear']}")
    print(f"  IS Sharpe: {best_params['sharpe']:.4f}")
    
    # 2) OOS 구간에서 검증
    print("\n" + "-"*100)
    print("2. OOS 구간에서 검증")
    print("-"*100)
    
    oos_result = test_regime_config(
        ret_raw=oos_ret,
        spx_close=oos_spx,
        bull=best_params['bull'],
        sideways=best_params['sideways'],
        bear=best_params['bear'],
    )
    
    print(f"\nOOS 구간 성과:")
    print(f"  Sharpe: {oos_result['sharpe']:.4f}")
    print(f"  연수익률: {oos_result['ann_return']*100:.2f}%")
    print(f"  연변동성: {oos_result['ann_vol']*100:.2f}%")
    print(f"  Max DD: {oos_result['max_dd']*100:.2f}%")
    
    # 3) 전체 구간 성과 (참고용)
    print("\n" + "-"*100)
    print("3. 전체 구간 성과 (참고용)")
    print("-"*100)
    
    full_result = test_regime_config(
        ret_raw=ret_raw,
        spx_close=spx_close,
        bull=best_params['bull'],
        sideways=best_params['sideways'],
        bear=best_params['bear'],
    )
    
    print(f"\n전체 구간 성과:")
    print(f"  Sharpe: {full_result['sharpe']:.4f}")
    print(f"  연수익률: {full_result['ann_return']*100:.2f}%")
    print(f"  연변동성: {full_result['ann_vol']*100:.2f}%")
    print(f"  Max DD: {full_result['max_dd']*100:.2f}%")
    
    return {
        "best_params": {
            "bull": float(best_params['bull']),
            "sideways": float(best_params['sideways']),
            "bear": float(best_params['bear']),
        },
        "is_performance": {
            "sharpe": float(best_params['sharpe']),
            "ann_return": float(best_params['ann_return']),
            "ann_vol": float(best_params['ann_vol']),
            "max_dd": float(best_params['max_dd']),
        },
        "oos_performance": {
            "sharpe": float(oos_result['sharpe']),
            "ann_return": float(oos_result['ann_return']),
            "ann_vol": float(oos_result['ann_vol']),
            "max_dd": float(oos_result['max_dd']),
        },
        "full_performance": {
            "sharpe": float(full_result['sharpe']),
            "ann_return": float(full_result['ann_return']),
            "ann_vol": float(full_result['ann_vol']),
            "max_dd": float(full_result['max_dd']),
        },
        "is_all_results": is_results.to_dict(orient="records"),
    }


def main():
    """메인 실행 함수"""
    print("Walk-forward 최적화 프레임워크")
    
    # 데이터 로드
    print("\n데이터 로딩...")
    spx_close, ret_v1_0 = load_data()
    print(f"  v1.0 수익률: {len(ret_v1_0)} 일")
    print(f"  SPX: {len(spx_close)} 일")
    print(f"  기간: {ret_v1_0.index[0].date()} ~ {ret_v1_0.index[-1].date()}")
    
    # Walk-forward 최적화
    # IS: 2018-02-01 ~ 2022-12-31 (약 5년)
    # OOS: 2023-01-01 ~ 2024-12-31 (약 2년)
    result = walkforward_optimize(
        ret_raw=ret_v1_0,
        spx_close=spx_close,
        is_start="2018-02-01",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2024-12-31",
    )
    
    # 결과 저장
    output_dir = Path("analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "walkforward_optimization.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
