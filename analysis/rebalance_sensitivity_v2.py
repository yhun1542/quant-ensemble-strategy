#!/usr/bin/env python3
"""
리밸런싱 민감도 실험 v2
Signal Smoothing 효과 검증
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.signal_smoothing import (
    get_rebalance_dates_with_offset,
    compute_signal_prices,
)
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


def calc_metrics(ret_daily: pd.Series) -> dict:
    """
    일간 수익률로부터 메트릭 계산
    
    Parameters
    ----------
    ret_daily : pd.Series
        일간 수익률
    
    Returns
    -------
    dict
        메트릭 딕셔너리
    """
    if len(ret_daily) < 2:
        return {}
    
    monthly_ret = ret_daily.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_ret) < 2:
        return {}
    
    ann_return = monthly_ret.mean() * 12
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    wealth = (1 + monthly_ret).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_dd = dd.min()
    
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret)
    
    return {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_months": len(monthly_ret),
    }


def simulate_rebalance_offset(
    ret_raw: pd.Series,
    spx_close: pd.Series,
    offset: int = 0,
    use_signal_smoothing: bool = False,
    window: int = 3,
) -> pd.Series:
    """
    리밸 오프셋을 적용한 시뮬레이션
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    spx_close : pd.Series
        S&P 500 종가
    offset : int
        리밸 날짜 오프셋 (0=첫날, 1=둘째날, ...)
    use_signal_smoothing : bool
        Signal Smoothing 사용 여부
    window : int
        Signal Smoothing 윈도우
    
    Returns
    -------
    pd.Series
        시뮬레이션 수익률
    """
    # 간단한 근사: 리밸 날짜를 offset만큼 밀어서 시뮬레이션
    # 실제로는 각 엔진의 백테스트를 다시 돌려야 하지만,
    # 여기서는 수익률 시리즈를 shift하는 방식으로 근사
    
    if offset == 0 and not use_signal_smoothing:
        # Baseline: 그대로 사용
        ret_sim = ret_raw.copy()
    elif offset > 0:
        # 리밸 날짜를 offset만큼 밀기
        # 간단한 근사: 수익률을 offset만큼 shift
        ret_sim = ret_raw.shift(offset).fillna(0)
    else:
        ret_sim = ret_raw.copy()
    
    # Signal Smoothing 적용 (간단한 근사)
    if use_signal_smoothing:
        rebal_dates = get_rebalance_dates_with_offset(ret_sim.index, offset=0)
        
        for rebal_date in rebal_dates:
            idx = ret_sim.index.get_loc(rebal_date)
            if idx + window <= len(ret_sim):
                # 리밸 날짜 이후 window일의 수익률 평균
                window_ret = ret_sim.iloc[idx:idx+window].mean()
                ret_sim.iloc[idx] = window_ret
    
    return ret_sim


def run_sensitivity_experiment(
    ret_raw: pd.Series,
    spx_close: pd.Series,
) -> pd.DataFrame:
    """
    리밸 민감도 실험 실행
    
    Parameters
    ----------
    ret_raw : pd.Series
        원본 수익률
    spx_close : pd.Series
        S&P 500 종가
    
    Returns
    -------
    pd.DataFrame
        실험 결과
    """
    print("="*100)
    print("리밸런싱 민감도 실험 v2")
    print("="*100)
    
    # 실험 시나리오
    scenarios = [
        # Baseline
        {"name": "Baseline (offset=0, no smoothing)", "offset": 0, "smoothing": False, "window": 1},
        
        # Case A: 3일 평균 시그널
        {"name": "Case A (offset=0, 3-day smoothing)", "offset": 0, "smoothing": True, "window": 3},
        
        # Case B: 오프셋 변경
        {"name": "Case B-1 (offset=1, no smoothing)", "offset": 1, "smoothing": False, "window": 1},
        {"name": "Case B-2 (offset=2, no smoothing)", "offset": 2, "smoothing": False, "window": 1},
        
        # Case C: 오프셋 + 스무딩
        {"name": "Case C-1 (offset=1, 3-day smoothing)", "offset": 1, "smoothing": True, "window": 3},
        {"name": "Case C-2 (offset=2, 3-day smoothing)", "offset": 2, "smoothing": True, "window": 3},
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
        
        # 시뮬레이션
        ret_sim = simulate_rebalance_offset(
            ret_raw=ret_raw,
            spx_close=spx_close,
            offset=scenario["offset"],
            use_signal_smoothing=scenario["smoothing"],
            window=scenario["window"],
        )
        
        # 리스크 레이어 적용 (v1.2 설정)
        risk_result = apply_risk_overlays(
            ret_raw=ret_sim,
            spx_close=spx_close,
            vol_cfg=VolTargetConfig(window_days=63, target_vol=0.15),
            dd_cfg=DrawdownConfig(warn_lvl=-0.05, cut_lvl=-0.10),
            regime_cfg=RegimeConfig(),
            regime_exp_cfg=RegimeExposureConfig(bull=1.0, sideways=0.5, bear=0.25),
        )
        
        ret_final = risk_result["ret_final"]
        
        # 메트릭 계산
        metrics = calc_metrics(ret_final)
        
        if metrics:
            print(f"  Sharpe: {metrics['sharpe']:.4f}")
            print(f"  Ann Return: {metrics['ann_return']*100:.2f}%")
            print(f"  Max DD: {metrics['max_dd']*100:.2f}%")
            
            results.append({
                "scenario": scenario["name"],
                "offset": scenario["offset"],
                "smoothing": scenario["smoothing"],
                "window": scenario["window"],
                **metrics,
            })
        else:
            print("  메트릭 계산 실패")
    
    df = pd.DataFrame(results)
    
    # 민감도 분석
    print("\n" + "="*100)
    print("민감도 분석")
    print("="*100)
    
    # Baseline vs Smoothing
    baseline_sharpe = df.loc[df["scenario"].str.contains("Baseline"), "sharpe"].iloc[0]
    
    # 오프셋별 Sharpe 분산
    no_smoothing = df.loc[~df["smoothing"], "sharpe"]
    with_smoothing = df.loc[df["smoothing"], "sharpe"]
    
    print(f"\nBaseline Sharpe: {baseline_sharpe:.4f}")
    print(f"\nNo Smoothing:")
    print(f"  Mean: {no_smoothing.mean():.4f}")
    print(f"  Std: {no_smoothing.std():.4f}")
    print(f"  CV: {no_smoothing.std() / no_smoothing.mean():.4f}")
    
    print(f"\nWith Smoothing:")
    print(f"  Mean: {with_smoothing.mean():.4f}")
    print(f"  Std: {with_smoothing.std():.4f}")
    print(f"  CV: {with_smoothing.std() / with_smoothing.mean():.4f}")
    
    # CV 개선율
    cv_improvement = (1 - (with_smoothing.std() / with_smoothing.mean()) / (no_smoothing.std() / no_smoothing.mean())) * 100
    print(f"\nCV 개선율: {cv_improvement:.2f}%")
    
    return df


def main():
    """메인 실행 함수"""
    print("리밸런싱 민감도 실험 v2")
    
    # 데이터 로드
    print("\n데이터 로딩...")
    spx_close, ret_v1_0 = load_data()
    print(f"  v1.0 수익률: {len(ret_v1_0)} 일")
    print(f"  SPX: {len(spx_close)} 일")
    
    # 실험 실행
    results_df = run_sensitivity_experiment(ret_v1_0, spx_close)
    
    # 결과 출력
    print("\n" + "="*100)
    print("전체 결과")
    print("="*100)
    print("\n" + results_df.to_string(index=False))
    
    # 결과 저장
    output_dir = Path("analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "rebalance_sensitivity_v2.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
