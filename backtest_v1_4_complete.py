#!/usr/bin/env python3
"""
v1.4 Complete Implementation: Full Engine-Level Backtest

This script runs the complete v1.4 strategy from scratch:
1. Load FV3c and ML9 engine results (with weights)
2. Create ensemble weights (60:40)
3. Apply regime filter + risk overlay
4. Apply Execution Smoothing v2
5. Calculate final metrics
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_with_execution_smoothing, ExecutionSmoothingConfig

print("="*100)
print("v1.4 Complete Implementation: Full Engine-Level Backtest")
print("="*100)

# 1. Load FV3c engine results
print("\n[1/6] Loading FV3c engine results...")
with open('results/ensemble_fv3c_ml9.json', 'r') as f:
    fv3c_data = json.load(f)

fv3c_weights_dict = fv3c_data['fv3c_weights']
fv3c_weights_by_date = {}
for date_str, weights_dict in fv3c_weights_dict.items():
    date = pd.Timestamp(date_str)
    fv3c_weights_by_date[date] = pd.Series(weights_dict)

print(f"  Loaded {len(fv3c_weights_by_date)} rebalance dates")

# 2. Load ML9 engine results
print("\n[2/6] Loading ML9 engine results...")
ml9_weights_dict = fv3c_data['ml9_weights']
ml9_weights_by_date = {}
for date_str, weights_dict in ml9_weights_dict.items():
    date = pd.Timestamp(date_str)
    ml9_weights_by_date[date] = pd.Series(weights_dict)

print(f"  Loaded {len(ml9_weights_by_date)} rebalance dates")

# 3. Create ensemble weights (60:40)
print("\n[3/6] Creating ensemble weights (60:40)...")
ensemble_weights_by_date = {}
all_dates = sorted(set(fv3c_weights_by_date.keys()) & set(ml9_weights_by_date.keys()))

for date in all_dates:
    fv3c_w = fv3c_weights_by_date[date]
    ml9_w = ml9_weights_by_date[date]
    
    # Combine all tickers
    all_tickers = fv3c_w.index.union(ml9_w.index)
    fv3c_full = fv3c_w.reindex(all_tickers).fillna(0.0)
    ml9_full = ml9_w.reindex(all_tickers).fillna(0.0)
    
    # 60:40 ensemble
    ensemble_w = 0.6 * fv3c_full + 0.4 * ml9_full
    ensemble_w = ensemble_w[ensemble_w.abs() > 1e-8]  # Drop near-zero
    
    ensemble_weights_by_date[date] = ensemble_w

print(f"  Created {len(ensemble_weights_by_date)} ensemble weight sets")

# 4. Load prices
print("\n[4/6] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")

# 5. Apply Execution Smoothing v2
print("\n[5/6] Applying Execution Smoothing v2...")
rebalance_dates = sorted(ensemble_weights_by_date.keys())

cfg = ExecutionSmoothingConfig(n_steps=2)
v1_4_returns = portfolio_returns_with_execution_smoothing(
    prices=prices,
    weights_by_date=ensemble_weights_by_date,
    rebalance_dates=rebalance_dates,
    cfg=cfg
)

print(f"  Calculated {len(v1_4_returns)} days of returns with Execution Smoothing")

# 6. Calculate metrics
print("\n[6/6] Calculating metrics...")

def calc_metrics(rets):
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    win_rate = (rets > 0).sum() / len(rets) if len(rets) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'max_dd': max_dd,
        'total_return': total,
        'win_rate': win_rate,
        'n_days': len(rets)
    }

v1_4_metrics = calc_metrics(v1_4_returns)

print("\n" + "="*100)
print("v1.4 Complete Implementation Results")
print("="*100)

print("\nv1.4 (FV3c + ML9 + Execution Smoothing v2):")
print(f"  Sharpe Ratio:   {v1_4_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {v1_4_metrics['annual_return']:.2%}")
print(f"  Annual Vol:     {v1_4_metrics['annual_vol']:.2%}")
print(f"  Max DD:         {v1_4_metrics['max_dd']:.2%}")
print(f"  Total Return:   {v1_4_metrics['total_return']:.2%}")
print(f"  Win Rate:       {v1_4_metrics['win_rate']:.2%}")
print(f"  Days:           {v1_4_metrics['n_days']}")

# Save results
results = {
    'strategy': 'v1.4_complete',
    'description': 'FV3c + ML9 (60:40) + Execution Smoothing v2 (split_steps=2)',
    'components': {
        'engines': ['FV3c', 'ML9'],
        'ensemble_weights': [0.6, 0.4],
        'execution_smoothing': 'v2 (split_steps=2)',
        'regime_filter': False,  # Not applied in this version
        'risk_overlay': False    # Not applied in this version
    },
    'metrics': v1_4_metrics,
    'daily_returns': {
        'index': [d.strftime('%Y-%m-%d') for d in v1_4_returns.index],
        'values': v1_4_returns.tolist()
    },
    'rebalance_dates': [d.strftime('%Y-%m-%d') for d in rebalance_dates]
}

with open('results/v1_4_complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to results/v1_4_complete_results.json")
print("\n" + "="*100)
print("v1.4 Complete Backtest Finished!")
print("="*100)
