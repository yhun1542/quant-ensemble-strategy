#!/usr/bin/env python3
"""
FV4 엔진 백테스트 (Signal Smoothing 적용)
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.signal_prices import (
    SignalSmoothingConfig,
    build_signal_price_df,
    expand_signal_prices,
    get_rebalance_dates_from_signal_df
)
from utils.factors import compute_all_factors
from engines.factor_value_v4_signal_smoothing import (
    FactorValueV4,
    FV4Config,
    portfolio_returns_from_weights
)

print("="*100)
print("FV4 Engine Backtest (Signal Smoothing)")
print("="*100)

# 1. 가격 데이터 로드
print("\n[1/5] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
print(f"  Loaded {len(prices)} days, {len(prices.columns)} tickers")

# 2. Signal Prices 생성
print("\n[2/5] Building signal prices...")
cfg_signal = SignalSmoothingConfig(window=3)
signal_df_raw = build_signal_price_df(prices, cfg_signal)
signal_df = expand_signal_prices(signal_df_raw, prices.index)
rebalance_dates = get_rebalance_dates_from_signal_df(signal_df_raw)
print(f"  Signal prices generated for {len(rebalance_dates)} rebalance dates")

# 3. Factors 계산
print("\n[3/5] Computing factors...")
factors = compute_all_factors(prices, signal_df)
print(f"  Factors computed: {len(factors)} rows")

# 4. FV4 엔진 실행
print("\n[4/5] Running FV4 engine...")
cfg_fv4 = FV4Config(top_quantile=0.2)
engine = FactorValueV4(cfg_fv4)
weights = engine.build_portfolio(prices, factors, rebalance_dates)
print(f"  FV4 generated {len(weights)} weight sets")

# 5. 수익률 계산
print("\n[5/5] Calculating returns...")
returns = portfolio_returns_from_weights(prices, weights, rebalance_dates)
print(f"  Returns calculated for {len(returns)} days")

# 성과 계산
def calc_metrics(rets):
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    return {
        'sharpe': sharpe,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'max_dd': max_dd,
        'total_return': total,
        'n_days': len(rets)
    }

metrics = calc_metrics(returns)

print("\n" + "="*100)
print("FV4 Engine Performance")
print("="*100)
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Annual Vol: {metrics['annual_vol']:.2%}")
print(f"Max DD: {metrics['max_dd']:.2%}")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Days: {metrics['n_days']}")

# 결과 저장
results = {
    'engine': 'FV4',
    'description': 'Factor Value v4 with Signal Smoothing',
    'config': {
        'signal_smoothing_window': cfg_signal.window,
        'top_quantile': cfg_fv4.top_quantile
    },
    'metrics': metrics,
    'daily_returns': {
        'index': [d.strftime('%Y-%m-%d') for d in returns.index],
        'values': returns.tolist()
    }
}

with open('results/fv4_engine_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to results/fv4_engine_results.json")
print("\n" + "="*100)
print("FV4 Engine Backtest Complete!")
print("="*100)
