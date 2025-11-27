#!/usr/bin/env python3
"""
앙상블 가중치 최적화
- Grid search: FV3c vs ML9 가중치
- 목표: Sharpe 1.2+, MaxDD < -10%
"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def calc_metrics(returns: pd.Series) -> Dict[str, float]:
    """성과 지표 계산"""
    returns = returns.fillna(0.0)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else 0.0
    annual_return = mean_ret * TRADING_DAYS
    annual_vol = std_ret * np.sqrt(TRADING_DAYS)
    
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
        "num_trades": len(returns)
    }


def main():
    print("=" * 100)
    print("앙상블 가중치 최적화 (Grid Search)")
    print("=" * 100)
    
    # 개별 엔진 결과 로드
    fv3c_path = "engine_results/factor_value_v3c_dynamic_oos.json"
    ml9_path = "engine_results/ml_xgboost_v9_ranking_oos.json"
    
    with open(fv3c_path) as f:
        fv3c_data = json.load(f)
    
    with open(ml9_path) as f:
        ml9_data = json.load(f)
    
    # Daily returns 추출
    fv3c_returns = pd.Series({
        pd.Timestamp(r["date"]): r["ret"]
        for r in fv3c_data["daily_returns"]
    }).sort_index()
    
    ml9_returns = pd.Series({
        pd.Timestamp(r["date"]): r["ret"]
        for r in ml9_data["daily_returns"]
    }).sort_index()
    
    # 공통 날짜
    common_dates = fv3c_returns.index.intersection(ml9_returns.index)
    fv3c_ret_common = fv3c_returns.loc[common_dates]
    ml9_ret_common = ml9_returns.loc[common_dates]
    
    print(f"\n공통 날짜 수: {len(common_dates)}")
    print(f"기간: {common_dates[0].date()} ~ {common_dates[-1].date()}")
    
    # Grid search
    print("\n" + "=" * 100)
    print("Grid Search 결과")
    print("=" * 100)
    
    weight_fv3c_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for w_fv3c in weight_fv3c_list:
        w_ml9 = 1.0 - w_fv3c
        
        # 앙상블 수익률
        ensemble_ret = w_fv3c * fv3c_ret_common + w_ml9 * ml9_ret_common
        
        # 성과 계산
        metrics = calc_metrics(ensemble_ret)
        
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
    
    # 2. MaxDD 최소화 (절대값)
    best_maxdd = min(results, key=lambda x: abs(x["max_drawdown"]))
    print(f"\n2. MaxDD 최소화:")
    print(f"   가중치: FV3c {best_maxdd['weight_fv3c']:.0%}, ML9 {best_maxdd['weight_ml9']:.0%}")
    print(f"   Sharpe: {best_maxdd['sharpe']:.4f}")
    print(f"   Return: {best_maxdd['annual_return']*100:.2f}%")
    print(f"   MaxDD: {best_maxdd['max_drawdown']*100:.2f}%")
    
    # 3. Sharpe 1.2+ 달성 가능한 가중치
    target_sharpe = 1.2
    target_maxdd = -0.10
    
    feasible = [r for r in results if r["sharpe"] >= target_sharpe and r["max_drawdown"] >= target_maxdd]
    
    print(f"\n3. 목표 달성 가중치 (Sharpe >= {target_sharpe}, MaxDD >= {target_maxdd*100:.0f}%):")
    if feasible:
        print(f"   ✅ {len(feasible)}개 가중치 조합이 목표 달성!")
        for r in feasible:
            print(f"   - FV3c {r['weight_fv3c']:.0%}, ML9 {r['weight_ml9']:.0%}: "
                  f"Sharpe {r['sharpe']:.4f}, MaxDD {r['max_drawdown']*100:.2f}%")
    else:
        print(f"   ❌ 목표 달성 가중치 없음")
        
        # 가장 근접한 가중치
        closest = min(results, key=lambda x: abs(x["sharpe"] - target_sharpe))
        print(f"\n   가장 근접한 가중치:")
        print(f"   - FV3c {closest['weight_fv3c']:.0%}, ML9 {closest['weight_ml9']:.0%}")
        print(f"   - Sharpe: {closest['sharpe']:.4f} (Gap: {target_sharpe - closest['sharpe']:.2f})")
        print(f"   - MaxDD: {closest['max_drawdown']*100:.2f}% (Gap: {(target_maxdd - closest['max_drawdown'])*100:.2f}%)")
    
    # 결과 저장
    output_path = Path("engine_results/ensemble_weight_optimization.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "best_sharpe": best_sharpe,
            "best_maxdd": best_maxdd,
            "feasible": feasible,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 결과 저장: {output_path}")
    
    # 추천
    print("\n" + "=" * 100)
    print("추천 가중치")
    print("=" * 100)
    
    if feasible:
        # 목표 달성 중 Sharpe 최대
        recommended = max(feasible, key=lambda x: x["sharpe"])
    else:
        # 목표 미달성 시 Sharpe 최대
        recommended = best_sharpe
    
    print(f"\n추천: FV3c {recommended['weight_fv3c']:.0%}, ML9 {recommended['weight_ml9']:.0%}")
    print(f"  - Sharpe: {recommended['sharpe']:.4f}")
    print(f"  - Annual Return: {recommended['annual_return']*100:.2f}%")
    print(f"  - Annual Vol: {recommended['annual_volatility']*100:.2f}%")
    print(f"  - MaxDD: {recommended['max_drawdown']*100:.2f}%")
    print(f"  - Win Rate: {recommended['win_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
