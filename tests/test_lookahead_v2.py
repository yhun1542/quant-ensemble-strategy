#!/usr/bin/env python3
"""
룩어헤드 바이어스 테스트 - v2 엔진용
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from engines.momentum_cs_v2_fixed import (
    MomentumCSEngineV2,
    MomentumCSEngineV2Config,
    load_price_data,
    get_monthly_rebalance_dates,
    portfolio_returns_from_weights,
    calc_monthly_metrics
)


def test_horizon_extension():
    """Horizon 변경 테스트"""
    print("="*100)
    print("Horizon 변경 테스트 (v2 엔진)")
    print("="*100)
    
    universe_30 = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
    ]
    
    prices = load_price_data("data/price_data_sp500.csv", universe_30)
    rebalance_dates = get_monthly_rebalance_dates(prices.index)
    
    # 원본 (252일)
    engine_252 = MomentumCSEngineV2(MomentumCSEngineV2Config(lookback_long=252))
    weights_252 = engine_252.build_portfolio(prices, rebalance_dates)
    ret_252 = portfolio_returns_from_weights(prices, weights_252, rebalance_dates)
    metrics_252 = calc_monthly_metrics(ret_252)
    
    print(f"\n252일 Horizon Sharpe: {metrics_252['sharpe']:.4f}")
    
    # 확장 (504일)
    engine_504 = MomentumCSEngineV2(MomentumCSEngineV2Config(lookback_long=504))
    weights_504 = engine_504.build_portfolio(prices, rebalance_dates)
    ret_504 = portfolio_returns_from_weights(prices, weights_504, rebalance_dates)
    metrics_504 = calc_monthly_metrics(ret_504)
    
    print(f"504일 Horizon Sharpe: {metrics_504['sharpe']:.4f}")
    
    change_pct = (metrics_504['sharpe'] - metrics_252['sharpe']) / metrics_252['sharpe'] * 100
    print(f"\n변화: {change_pct:+.1f}%")
    
    if change_pct < -20:
        print("✅ PASS - Horizon 확장 시 성능 크게 저하")
    elif change_pct < -5:
        print("⚠️  CAUTION - Horizon 확장 시 성능 다소 저하")
    else:
        print("❌ FAIL - Horizon 확장해도 성능 유지")


def test_manual_verification():
    """수동 검증 테스트"""
    print("\n" + "="*100)
    print("수동 검증 테스트 (v2 엔진)")
    print("="*100)
    
    universe_30 = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
    ]
    
    prices = load_price_data("data/price_data_sp500.csv", universe_30)
    engine = MomentumCSEngineV2()
    factors = engine.compute_momentum_factors(prices)
    
    # 샘플 선택
    np.random.seed(42)
    available_dates = factors.index.get_level_values(0).unique()
    sample_dates = np.random.choice(available_dates[252:], size=5, replace=False)
    
    print(f"\n검증 샘플:")
    verification_results = []
    
    for date in sample_dates:
        ticker = np.random.choice(universe_30)
        
        try:
            # 엔진에서 계산된 값
            factor_value = factors.loc[(date, ticker), 'mom_252_ex_21']
            
            # 수동 재계산
            price_hist = prices.loc[:date, ticker]
            if len(price_hist) >= 252:
                p_t_minus_21 = price_hist.iloc[-22]  # shift(21)은 현재 포함 22개 필요
                p_t_minus_252 = price_hist.iloc[-253]  # shift(252)는 현재 포함 253개 필요
                manual_value = p_t_minus_21 / p_t_minus_252 - 1.0
                
                diff = abs(factor_value - manual_value)
                match = diff < 1e-6
                
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                print(f"  - {date_str} | {ticker:5s} | 엔진: {factor_value:8.4f} | 수동: {manual_value:8.4f} | 차이: {diff:.2e} | {'✓' if match else '✗'}")
                
                verification_results.append({
                    'date': date,
                    'ticker': ticker,
                    'engine_value': factor_value,
                    'manual_value': manual_value,
                    'diff': diff,
                    'match': match
                })
        except Exception as e:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            print(f"  - {date_str} | {ticker:5s} | 오류: {str(e)}")
    
    if verification_results:
        match_rate = sum(r['match'] for r in verification_results) / len(verification_results)
        avg_diff = np.mean([r['diff'] for r in verification_results])
        
        print(f"\n검증 결과:")
        print(f"  - 일치율: {match_rate*100:.1f}%")
        print(f"  - 평균 차이: {avg_diff:.2e}")
        
        if match_rate >= 0.95:
            print("✅ PASS - 수동 계산과 일치")
        elif match_rate >= 0.8:
            print("⚠️  CAUTION - 일부 불일치")
        else:
            print("❌ FAIL - 수동 계산과 불일치")


if __name__ == "__main__":
    test_horizon_extension()
    test_manual_verification()
