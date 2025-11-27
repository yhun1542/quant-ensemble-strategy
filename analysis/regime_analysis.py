#!/usr/bin/env python3
"""
레짐 분석 및 파라미터 튜닝
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.regime import compute_spx_regime, RegimeConfig
from utils.risk_overlay import RegimeExposureConfig, apply_risk_overlays, VolTargetConfig, DrawdownConfig


def load_data():
    """데이터 로드"""
    # S&P 500
    spx_df = pd.read_csv("data/spx_close.csv", index_col=0, parse_dates=True)
    spx_close = spx_df["SPX"]
    
    # v1.0 앙상블 수익률
    with open("results/ensemble_v1_2_backtest.json", "r") as f:
        data = json.load(f)
    
    ret_v1_0 = pd.Series(
        data["daily_returns"]["v1_0"]["values"],
        index=pd.to_datetime(data["daily_returns"]["v1_0"]["index"]),
        name="v1_0",
    )
    
    return spx_close, ret_v1_0


def analyze_regime_distribution(spx_close: pd.Series):
    """레짐 분포 분석"""
    print("="*100)
    print("레짐 분포 분석")
    print("="*100)
    
    regime = compute_spx_regime(spx_close)
    
    # 전체 분포
    print("\n전체 기간 레짐 분포:")
    print(regime.value_counts())
    print(f"\n비율:")
    print(regime.value_counts(normalize=True))
    
    # 시기별 분포
    print("\n\n시기별 레짐 분포:")
    for year in range(2021, 2026):
        year_regime = regime.loc[str(year)]
        if len(year_regime) > 0:
            print(f"\n{year}년:")
            print(year_regime.value_counts())
    
    return regime


def test_regime_exposures(ret_raw: pd.Series, spx_close: pd.Series):
    """다양한 레짐 익스포저 설정 테스트"""
    print("\n" + "="*100)
    print("레짐 익스포저 파라미터 테스트")
    print("="*100)
    
    # 테스트할 설정들
    configs = [
        {"bull": 1.0, "sideways": 0.5, "bear": 0.0, "name": "보수적 (bear=0.0)"},
        {"bull": 1.0, "sideways": 0.5, "bear": 0.25, "name": "중립 (bear=0.25)"},
        {"bull": 1.0, "sideways": 0.75, "bear": 0.5, "name": "공격적 (bear=0.5)"},
        {"bull": 1.0, "sideways": 1.0, "bear": 1.0, "name": "레짐 필터 없음"},
    ]
    
    results = []
    
    for cfg in configs:
        regime_exp_cfg = RegimeExposureConfig(
            bull=cfg["bull"],
            sideways=cfg["sideways"],
            bear=cfg["bear"],
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
        
        # 월간 메트릭
        monthly_ret = ret_final.resample("M").apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly_ret) >= 2:
            ann_return = monthly_ret.mean() * 12
            ann_vol = monthly_ret.std() * np.sqrt(12)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            wealth = (1 + monthly_ret).cumprod()
            running_max = wealth.cummax()
            dd = wealth / running_max - 1
            max_dd = dd.min()
        else:
            ann_return = 0
            ann_vol = 0
            sharpe = 0
            max_dd = 0
        
        results.append({
            "name": cfg["name"],
            "bull": cfg["bull"],
            "sideways": cfg["sideways"],
            "bear": cfg["bear"],
            "sharpe": sharpe,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "max_dd": max_dd,
        })
    
    # 결과 출력
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # 최적 설정 찾기
    best_idx = df["sharpe"].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n최적 설정: {best['name']}")
    print(f"  - Sharpe: {best['sharpe']:.4f}")
    print(f"  - 연수익률: {best['ann_return']*100:.2f}%")
    print(f"  - Max DD: {best['max_dd']*100:.2f}%")
    
    return df


def main():
    """메인 실행 함수"""
    print("레짐 분석 및 파라미터 튜닝")
    
    # 데이터 로드
    spx_close, ret_v1_0 = load_data()
    
    # 1) 레짐 분포 분석
    regime = analyze_regime_distribution(spx_close)
    
    # 2) 레짐 익스포저 테스트
    results_df = test_regime_exposures(ret_v1_0, spx_close)
    
    # 3) 결과 저장
    output_dir = Path("analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "regime_exposure_tuning.csv", index=False)
    print(f"\n✅ 결과 저장: {output_dir / 'regime_exposure_tuning.csv'}")


if __name__ == "__main__":
    main()
