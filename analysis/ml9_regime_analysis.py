"""
ML9 Market Regime Analysis

목표: ML9의 성과가 특정 시장 환경에 의존하는지 분석
- 2021-2024 기간을 시장 레짐별로 분할
- 각 레짐에서의 성과 측정
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_ml9_results():
    """ML9 결과 로딩"""
    # v2.1 결과에서 일간 수익률 추출
    with open("/home/ubuntu/quant-ensemble-strategy/results/ensemble_fv3c_ml9.json", "r") as f:
        data = json.load(f)
    
    # ML9 weights 로딩
    ml9_weights = {}
    for date_str, weights_dict in data["ml9_weights"].items():
        if weights_dict:
            date = pd.to_datetime(date_str).tz_localize(None)
            ml9_weights[date] = pd.Series(weights_dict)
    
    return ml9_weights


def load_price_data():
    """가격 데이터 로딩"""
    prices = pd.read_csv("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp500.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date")
    prices.index = prices.index.tz_localize(None)
    
    TICKERS = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
    ]
    available_tickers = [t for t in TICKERS if t in prices.columns]
    prices = prices[available_tickers]
    
    return prices


def portfolio_returns(prices: pd.DataFrame, weights_by_date: dict) -> pd.Series:
    """포트폴리오 수익률 계산"""
    returns_daily = prices.pct_change()
    portfolio_returns = []
    
    weights_by_date_only = {}
    for dt, w in weights_by_date.items():
        date_only = dt.date()
        weights_by_date_only[date_only] = w
    
    current_weights = None
    
    for date in prices.index:
        date_only = date.date()
        
        if date_only in weights_by_date_only:
            current_weights = weights_by_date_only[date_only]
        
        if current_weights is None:
            continue
        
        ret_cross = returns_daily.loc[date]
        common_tickers = current_weights.index.intersection(ret_cross.index)
        
        if len(common_tickers) > 0:
            w = current_weights[common_tickers]
            r = ret_cross[common_tickers]
            port_ret = (w * r).sum()
            portfolio_returns.append({"date": date, "return": port_ret})
    
    df_ret = pd.DataFrame(portfolio_returns)
    if df_ret.empty:
        return pd.Series(dtype=float)
    
    df_ret = df_ret.set_index("date")
    return df_ret["return"]


def calculate_metrics(returns: pd.Series, name: str = "Strategy") -> dict:
    """성과 지표 계산"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns)
    
    metrics = {
        "name": name,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "days": len(returns),
    }
    
    return metrics


def main():
    print("="*100)
    print("ML9 MARKET REGIME ANALYSIS")
    print("="*100)
    
    # 1. 데이터 로딩
    print("\n1. Loading data...")
    prices = load_price_data()
    ml9_weights = load_ml9_results()
    print(f"Prices: {len(prices)} days")
    print(f"ML9 weights: {len(ml9_weights)} rebalance dates")
    
    # 2. 포트폴리오 수익률 계산
    print("\n2. Calculating portfolio returns...")
    returns = portfolio_returns(prices, ml9_weights)
    print(f"Returns: {len(returns)} days")
    
    # 3. 시장 레짐 정의
    print("\n" + "="*100)
    print("3. MARKET REGIME ANALYSIS")
    print("="*100)
    
    regimes = [
        {"name": "2021 H2 - Post-COVID Rally", "start": "2021-10-01", "end": "2021-12-31"},
        {"name": "2022 - Bear Market", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "2023 - AI Boom", "start": "2023-01-01", "end": "2023-12-31"},
        {"name": "2024 - Consolidation", "start": "2024-01-01", "end": "2024-12-31"},
    ]
    
    regime_results = []
    
    for regime in regimes:
        print(f"\n--- {regime['name']} ({regime['start']} ~ {regime['end']}) ---")
        
        regime_returns = returns.loc[regime['start']:regime['end']]
        
        if len(regime_returns) == 0:
            print("⚠️ No data")
            continue
        
        metrics = calculate_metrics(regime_returns, regime['name'])
        
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Vol: {metrics['annual_vol']:.2%}")
        print(f"Max DD: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Days: {metrics['days']}")
        
        regime_results.append({
            **regime,
            **metrics
        })
    
    # 4. 결과 저장
    print("\n" + "="*100)
    print("4. SAVING RESULTS")
    print("="*100)
    
    results = {
        "analysis": "ML9_Market_Regime_Analysis",
        "period": "2021-10-01 to 2024-12-31",
        "regimes": regime_results,
        "summary": {
            "best_regime": max(regime_results, key=lambda x: x['sharpe_ratio'])['name'],
            "worst_regime": min(regime_results, key=lambda x: x['sharpe_ratio'])['name'],
            "sharpe_range": [
                min(r['sharpe_ratio'] for r in regime_results),
                max(r['sharpe_ratio'] for r in regime_results)
            ]
        }
    }
    
    out_path = Path("/home/ubuntu/quant-ensemble-strategy/results/ml9_regime_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {out_path}")
    
    # 5. 요약
    print("\n" + "="*100)
    print("5. SUMMARY")
    print("="*100)
    
    print(f"\nBest Regime: {results['summary']['best_regime']}")
    print(f"Worst Regime: {results['summary']['worst_regime']}")
    print(f"Sharpe Range: {results['summary']['sharpe_range'][0]:.2f} ~ {results['summary']['sharpe_range'][1]:.2f}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
