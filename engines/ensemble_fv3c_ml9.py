#!/usr/bin/env python3
"""
Ensemble: Factor Value v3c + ML XGBoost v9
- Equal weight (50:50)
- Out-of-Sample backtest
- Correlation analysis
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class EnsembleFV3cML9:
    """
    Ensemble: Factor Value v3c + ML XGBoost v9
    """
    
    def __init__(self, fv3c_returns_path: str, ml9_returns_path: str,
                 weight_fv3c: float = 0.5, weight_ml9: float = 0.5):
        # 개별 엔진 결과 로드
        with open(fv3c_returns_path) as f:
            fv3c_data = json.load(f)
        
        with open(ml9_returns_path) as f:
            ml9_data = json.load(f)
        
        # Daily returns 추출
        self.fv3c_returns = pd.Series({
            pd.Timestamp(r["date"]): r["ret"]
            for r in fv3c_data["daily_returns"]
        }).sort_index()
        
        self.ml9_returns = pd.Series({
            pd.Timestamp(r["date"]): r["ret"]
            for r in ml9_data["daily_returns"]
        }).sort_index()
        
        self.weight_fv3c = weight_fv3c
        self.weight_ml9 = weight_ml9
        
        # 개별 엔진 성과
        self.fv3c_metrics = fv3c_data["overall"]
        self.ml9_metrics = ml9_data["overall"]
        
    def _calc_metrics(self, returns: pd.Series) -> PerformanceMetrics:
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
        
        return PerformanceMetrics(
            sharpe=float(sharpe),
            annual_return=float(annual_return),
            annual_volatility=float(annual_vol),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            num_trades=len(returns)
        )
    
    def run_ensemble(self) -> Dict[str, Any]:
        """앙상블 백테스트"""
        # 공통 날짜 찾기
        common_dates = self.fv3c_returns.index.intersection(self.ml9_returns.index)
        
        if len(common_dates) == 0:
            raise RuntimeError("No common dates between engines")
        
        print(f"공통 날짜 수: {len(common_dates)}")
        print(f"기간: {common_dates[0].date()} ~ {common_dates[-1].date()}")
        
        # 앙상블 수익률 계산
        ensemble_returns = (
            self.weight_fv3c * self.fv3c_returns.loc[common_dates] +
            self.weight_ml9 * self.ml9_returns.loc[common_dates]
        )
        
        # 상관관계 분석
        fv3c_ret_common = self.fv3c_returns.loc[common_dates]
        ml9_ret_common = self.ml9_returns.loc[common_dates]
        correlation = fv3c_ret_common.corr(ml9_ret_common)
        
        # 성과 계산
        ensemble_metrics = self._calc_metrics(ensemble_returns)
        
        return {
            "ensemble": asdict(ensemble_metrics),
            "fv3c": self.fv3c_metrics,
            "ml9": self.ml9_metrics,
            "correlation": float(correlation),
            "weights": {
                "fv3c": self.weight_fv3c,
                "ml9": self.weight_ml9
            },
            "daily_returns": [
                {"date": d.strftime("%Y-%m-%d"), "ret": float(r)}
                for d, r in ensemble_returns.items()
            ],
        }


def main():
    print("=" * 100)
    print("Ensemble: Factor Value v3c + ML XGBoost v9")
    print("=" * 100)
    
    # 개별 엔진 결과 경로
    fv3c_path = "engine_results/factor_value_v3c_dynamic_oos.json"
    ml9_path = "engine_results/ml_xgboost_v9_ranking_oos.json"
    
    # 파일 존재 확인
    if not Path(fv3c_path).exists():
        print(f"❌ {fv3c_path} not found")
        print("Factor Value v3c를 먼저 실행해주세요")
        return
    
    if not Path(ml9_path).exists():
        print(f"❌ {ml9_path} not found")
        print("ML XGBoost v9를 먼저 실행해주세요")
        return
    
    # 앙상블 실행
    ensemble = EnsembleFV3cML9(fv3c_path, ml9_path)
    result = ensemble.run_ensemble()
    
    # 저장
    output_path = Path("engine_results/ensemble_fv3c_ml9_oos.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 결과 출력
    print("\n" + "=" * 100)
    print("개별 엔진 성과")
    print("=" * 100)
    
    print("\nFactor Value v3c:")
    print(f"  Sharpe: {result['fv3c']['sharpe']:.4f}")
    print(f"  Annual Return: {result['fv3c']['annual_return']*100:.2f}%")
    print(f"  Annual Vol: {result['fv3c']['annual_volatility']*100:.2f}%")
    print(f"  Max DD: {result['fv3c']['max_drawdown']*100:.2f}%")
    
    print("\nML XGBoost v9:")
    print(f"  Sharpe: {result['ml9']['sharpe']:.4f}")
    print(f"  Annual Return: {result['ml9']['annual_return']*100:.2f}%")
    print(f"  Annual Vol: {result['ml9']['annual_volatility']*100:.2f}%")
    print(f"  Max DD: {result['ml9']['max_drawdown']*100:.2f}%")
    
    print("\n" + "=" * 100)
    print("앙상블 성과 (50:50)")
    print("=" * 100)
    
    print(f"\nSharpe Ratio: {result['ensemble']['sharpe']:.4f}")
    print(f"Annual Return: {result['ensemble']['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {result['ensemble']['annual_volatility']*100:.2f}%")
    print(f"Max Drawdown: {result['ensemble']['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {result['ensemble']['win_rate']*100:.2f}%")
    
    print("\n" + "=" * 100)
    print("다양성 분석")
    print("=" * 100)
    
    print(f"\n상관관계: {result['correlation']:.4f}")
    
    if result['correlation'] < 0.5:
        print("✅ 낮은 상관관계 → 다양성 확보")
    elif result['correlation'] < 0.7:
        print("⚠️ 중간 상관관계 → 일부 다양성")
    else:
        print("❌ 높은 상관관계 → 다양성 부족")
    
    # 목표 달성 여부
    print("\n" + "=" * 100)
    print("목표 달성 여부")
    print("=" * 100)
    
    target_sharpe = 1.2
    target_maxdd = -0.10
    
    print(f"\n목표 Sharpe: {target_sharpe:.2f}")
    print(f"앙상블 Sharpe: {result['ensemble']['sharpe']:.4f}")
    
    if result['ensemble']['sharpe'] >= target_sharpe:
        print("✅ Sharpe 목표 달성!")
    else:
        gap = target_sharpe - result['ensemble']['sharpe']
        print(f"❌ Sharpe 목표 미달 (Gap: {gap:.2f})")
    
    print(f"\n목표 MaxDD: {target_maxdd*100:.0f}%")
    print(f"앙상블 MaxDD: {result['ensemble']['max_drawdown']*100:.2f}%")
    
    if result['ensemble']['max_drawdown'] >= target_maxdd:
        print("✅ MaxDD 목표 달성!")
    else:
        print("❌ MaxDD 목표 미달")
    
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
