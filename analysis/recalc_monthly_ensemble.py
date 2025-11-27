#!/usr/bin/env python3
"""
월간 수익률 기반 앙상블 재계산
- 일간 수익률 → 월간 수익률 변환
- 월간 Sharpe 계산
- 가중치 최적화
"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def calc_monthly_metrics(returns: pd.Series) -> Dict[str, float]:
    """월간 수익률 기반 성과 지표 계산"""
    returns = returns.fillna(0.0)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe = (mean_ret * MONTHS_PER_YEAR) / (std_ret * np.sqrt(MONTHS_PER_YEAR)) if std_ret > 0 else 0.0
    annual_return = mean_ret * MONTHS_PER_YEAR
    annual_vol = std_ret * np.sqrt(MONTHS_PER_YEAR)
    
    cum_ret = (1.0 + returns).cumprod()
    peak = cum_ret.cummax()
    dd = cum_ret / peak - 1.0
    max_dd = dd.min()
    
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
    
    return {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "num_periods": len(returns)
    }


def daily_to_monthly(daily_returns: pd.Series) -> pd.Series:
    """일간 수익률 → 월간 수익률 변환"""
    # 월별 그룹화
    monthly_returns = (1.0 + daily_returns).groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).prod() - 1.0
    
    # Index를 월말 날짜로 변환
    monthly_returns.index = pd.to_datetime([
        f"{year}-{month:02d}-01" 
        for year, month in monthly_returns.index
    ]) + pd.offsets.MonthEnd(0)
    
    return monthly_returns


def main():
    print("=" * 100)
    print("월간 수익률 기반 앙상블 재계산")
    print("=" * 100)
    
    # 개별 엔진 결과 로드
    fv3c_path = "engine_results/factor_value_v3c_dynamic_oos.json"
    ml9_path = "engine_results/ml_xgboost_v9_ranking_oos.json"
    
    with open(fv3c_path) as f:
        fv3c_data = json.load(f)
    
    with open(ml9_path) as f:
        ml9_data = json.load(f)
    
    # 일간 수익률 추출
    fv3c_daily = pd.Series({
        pd.Timestamp(r["date"]): r["ret"]
        for r in fv3c_data["daily_returns"]
    }).sort_index()
    
    ml9_daily = pd.Series({
        pd.Timestamp(r["date"]): r["ret"]
        for r in ml9_data["daily_returns"]
    }).sort_index()
    
    print(f"\n일간 수익률:")
    print(f"  FV3c: {len(fv3c_daily)}개")
    print(f"  ML9: {len(ml9_daily)}개")
    
    # 월간 수익률 변환
    print("\n월간 수익률 변환 중...")
    fv3c_monthly = daily_to_monthly(fv3c_daily)
    ml9_monthly = daily_to_monthly(ml9_daily)
    
    print(f"\n월간 수익률:")
    print(f"  FV3c: {len(fv3c_monthly)}개")
    print(f"  ML9: {len(ml9_monthly)}개")
    print(f"  기간: {fv3c_monthly.index[0].date()} ~ {fv3c_monthly.index[-1].date()}")
    
    # 공통 월 찾기
    common_months = fv3c_monthly.index.intersection(ml9_monthly.index)
    fv3c_ret = fv3c_monthly.loc[common_months]
    ml9_ret = ml9_monthly.loc[common_months]
    
    print(f"\n공통 월: {len(common_months)}개")
    
    # 개별 엔진 성과 (월간 기준)
    print("\n" + "=" * 100)
    print("개별 엔진 성과 (월간 수익률 기준)")
    print("=" * 100)
    
    fv3c_metrics = calc_monthly_metrics(fv3c_ret)
    ml9_metrics = calc_monthly_metrics(ml9_ret)
    
    print("\nFactor Value v3c:")
    print(f"  Sharpe: {fv3c_metrics['sharpe']:.4f}")
    print(f"  Annual Return: {fv3c_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Vol: {fv3c_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max DD: {fv3c_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {fv3c_metrics['win_rate']*100:.2f}%")
    
    print("\nML XGBoost v9:")
    print(f"  Sharpe: {ml9_metrics['sharpe']:.4f}")
    print(f"  Annual Return: {ml9_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Vol: {ml9_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max DD: {ml9_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {ml9_metrics['win_rate']*100:.2f}%")
    
    # 상관관계
    correlation = fv3c_ret.corr(ml9_ret)
    print(f"\n상관관계 (월간): {correlation:.4f}")
    
    # 가중치 최적화
    print("\n" + "=" * 100)
    print("가중치 최적화 (월간 수익률 기준)")
    print("=" * 100)
    
    weight_fv3c_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    for w_fv3c in weight_fv3c_list:
        w_ml9 = 1.0 - w_fv3c
        
        # 앙상블 월간 수익률
        ensemble_ret = w_fv3c * fv3c_ret + w_ml9 * ml9_ret
        
        # 성과 계산
        metrics = calc_monthly_metrics(ensemble_ret)
        
        results.append({
            "weight_fv3c": w_fv3c,
            "weight_ml9": w_ml9,
            **metrics
        })
    
    # 결과 출력
    print("\n가중치별 성과:")
    print("-" * 100)
    print(f"{'FV3c':>6} {'ML9':>6} {'Sharpe':>8} {'Return':>8} {'Vol':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['weight_fv3c']:>6.0%} {r['weight_ml9']:>6.0%} "
              f"{r['sharpe']:>8.4f} {r['annual_return']*100:>7.2f}% "
              f"{r['annual_volatility']*100:>7.2f}% {r['max_drawdown']*100:>7.2f}% "
              f"{r['win_rate']*100:>7.2f}%")
    
    # 최적 가중치 찾기
    print("\n" + "=" * 100)
    print("최적 가중치 분석")
    print("=" * 100)
    
    # 1. Sharpe 최대화
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    print(f"\n1. Sharpe 최대화:")
    print(f"   가중치: FV3c {best_sharpe['weight_fv3c']:.0%}, ML9 {best_sharpe['weight_ml9']:.0%}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.4f}")
    print(f"   Return: {best_sharpe['annual_return']*100:.2f}%")
    print(f"   MaxDD: {best_sharpe['max_drawdown']*100:.2f}%")
    
    # 2. 목표 달성 가중치
    target_sharpe = 1.2
    target_maxdd = -0.10
    
    feasible = [r for r in results if r["sharpe"] >= target_sharpe and r["max_drawdown"] >= target_maxdd]
    
    print(f"\n2. 목표 달성 가중치 (Sharpe >= {target_sharpe}, MaxDD >= {target_maxdd*100:.0f}%):")
    if feasible:
        print(f"   ✅ {len(feasible)}개 가중치 조합이 목표 달성!")
        for r in feasible:
            print(f"   - FV3c {r['weight_fv3c']:.0%}, ML9 {r['weight_ml9']:.0%}: "
                  f"Sharpe {r['sharpe']:.4f}, MaxDD {r['max_drawdown']*100:.2f}%")
    else:
        print(f"   ❌ 목표 달성 가중치 없음")
        closest = min(results, key=lambda x: abs(x["sharpe"] - target_sharpe))
        print(f"\n   가장 근접한 가중치:")
        print(f"   - FV3c {closest['weight_fv3c']:.0%}, ML9 {closest['weight_ml9']:.0%}")
        print(f"   - Sharpe: {closest['sharpe']:.4f} (Gap: {target_sharpe - closest['sharpe']:.2f})")
        print(f"   - MaxDD: {closest['max_drawdown']*100:.2f}%")
    
    # 일간 vs 월간 비교
    print("\n" + "=" * 100)
    print("일간 vs 월간 비교 (60:40 가중치)")
    print("=" * 100)
    
    # 60:40 가중치 찾기
    result_60_40 = [r for r in results if r["weight_fv3c"] == 0.6][0]
    
    print("\n일간 수익률 기준:")
    print("  Sharpe: 1.1201")
    print("  Annual Return: 13.37%")
    print("  Annual Vol: 11.93%")
    print("  MaxDD: -12.88%")
    
    print("\n월간 수익률 기준:")
    print(f"  Sharpe: {result_60_40['sharpe']:.4f}")
    print(f"  Annual Return: {result_60_40['annual_return']*100:.2f}%")
    print(f"  Annual Vol: {result_60_40['annual_volatility']*100:.2f}%")
    print(f"  MaxDD: {result_60_40['max_drawdown']*100:.2f}%")
    
    sharpe_improvement = ((result_60_40['sharpe'] / 1.1201) - 1) * 100
    print(f"\nSharpe 개선: {sharpe_improvement:+.1f}%")
    
    # 결과 저장
    output_path = Path("engine_results/ensemble_monthly_optimization.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "fv3c_monthly": fv3c_metrics,
            "ml9_monthly": ml9_metrics,
            "correlation_monthly": float(correlation),
            "results": results,
            "best_sharpe": best_sharpe,
            "feasible": feasible,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 결과 저장: {output_path}")
    
    # 추천
    print("\n" + "=" * 100)
    print("최종 추천")
    print("=" * 100)
    
    if feasible:
        recommended = max(feasible, key=lambda x: x["sharpe"])
        print(f"\n✅ 목표 달성! 추천 가중치: FV3c {recommended['weight_fv3c']:.0%}, ML9 {recommended['weight_ml9']:.0%}")
    else:
        recommended = best_sharpe
        print(f"\n⚠️ 목표 근접. 추천 가중치: FV3c {recommended['weight_fv3c']:.0%}, ML9 {recommended['weight_ml9']:.0%}")
    
    print(f"  - Sharpe: {recommended['sharpe']:.4f}")
    print(f"  - Annual Return: {recommended['annual_return']*100:.2f}%")
    print(f"  - Annual Vol: {recommended['annual_volatility']*100:.2f}%")
    print(f"  - MaxDD: {recommended['max_drawdown']*100:.2f}%")
    print(f"  - Win Rate: {recommended['win_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
