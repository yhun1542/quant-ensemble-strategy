#!/usr/bin/env python3
"""
v1.4 Complete Implementation: Long-Only Strategy with Execution Smoothing v2

Based on testing results:
- Long-Short strategy failed (-310% return)
- Long-Only strategy succeeded (Sharpe 2.14, 36% annual return)

Therefore, v1.4 uses Long-Only approach like v1.2.
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_with_execution_smoothing, ExecutionSmoothingConfig

print("="*100)
print("v1.4 Complete Implementation: Long-Only + Execution Smoothing v2")
print("="*100)

# 1. Load FV3c and ML9 weights
print("\n[1/5] Loading engine weights...")
with open('results/ensemble_fv3c_ml9.json', 'r') as f:
    data = json.load(f)

fv3c_weights_dict = data['fv3c_weights']
ml9_weights_dict = data['ml9_weights']

print(f"  FV3c: {len(fv3c_weights_dict)} dates")
print(f"  ML9: {len(ml9_weights_dict)} dates")

# 2. Load prices
print("\n[2/5] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
prices.index = prices.index.normalize()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")

# 3. Create Long-Only ensemble weights (60:40)
print("\n[3/5] Creating Long-Only ensemble weights (60:40)...")
ensemble_weights_by_date = {}
all_dates = sorted(set(fv3c_weights_dict.keys()) & set(ml9_weights_dict.keys()))

for date_str in all_dates:
    date = pd.Timestamp(date_str).normalize()
    
    fv3c_w = pd.Series(fv3c_weights_dict[date_str])
    ml9_w = pd.Series(ml9_weights_dict[date_str])
    
    # Combine all tickers
    all_tickers = fv3c_w.index.union(ml9_w.index)
    fv3c_full = fv3c_w.reindex(all_tickers).fillna(0.0)
    ml9_full = ml9_w.reindex(all_tickers).fillna(0.0)
    
    # 60:40 ensemble
    ensemble_w = 0.6 * fv3c_full + 0.4 * ml9_full
    
    # Long-Only: keep only positive weights
    long_only_w = ensemble_w[ensemble_w > 0]
    
    if len(long_only_w) > 0:
        # Normalize to sum=1.0
        long_only_w = long_only_w / long_only_w.sum()
        ensemble_weights_by_date[date] = long_only_w

print(f"  Created {len(ensemble_weights_by_date)} long-only weight sets")

# Debug: Check first weights
first_date = sorted(ensemble_weights_by_date.keys())[0]
print(f"\n  First ensemble date: {first_date}")
print(f"  Weights sum: {ensemble_weights_by_date[first_date].sum():.4f}")
print(f"  Number of positions: {len(ensemble_weights_by_date[first_date])}")

# 4. Apply Execution Smoothing v2
print("\n[4/5] Applying Execution Smoothing v2...")
rebalance_dates = sorted(ensemble_weights_by_date.keys())

cfg = ExecutionSmoothingConfig(n_steps=2)
v1_4_returns = portfolio_returns_with_execution_smoothing(
    prices=prices,
    weights_by_date=ensemble_weights_by_date,
    rebalance_dates=rebalance_dates,
    cfg=cfg
)

print(f"  Calculated {len(v1_4_returns)} days of returns with Execution Smoothing")

# 5. Calculate metrics
print("\n[5/5] Calculating metrics...")

def calc_metrics(rets):
    if len(rets) == 0:
        return {
            'sharpe': 0,
            'annual_return': 0,
            'annual_vol': 0,
            'max_dd': 0,
            'total_return': 0,
            'win_rate': 0,
            'n_days': 0
        }
    
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    
    if n_years > 0 and total > -1:
        ann_ret = (1 + total) ** (1/n_years) - 1
    else:
        ann_ret = np.nan
    
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 and not np.isnan(ann_ret) else np.nan
    
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
print("v1.4 Complete Implementation Results (Long-Only)")
print("="*100)

print("\nv1.4 (FV3c + ML9 Long-Only + Execution Smoothing v2):")
print(f"  Sharpe Ratio:   {v1_4_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {v1_4_metrics['annual_return']:.2%}")
print(f"  Annual Vol:     {v1_4_metrics['annual_vol']:.2%}")
print(f"  Max DD:         {v1_4_metrics['max_dd']:.2%}")
print(f"  Total Return:   {v1_4_metrics['total_return']:.2%}")
print(f"  Win Rate:       {v1_4_metrics['win_rate']:.2%}")
print(f"  Days:           {v1_4_metrics['n_days']}")

# Save results
results = {
    'strategy': 'v1.4_long_only',
    'description': 'FV3c + ML9 (60:40) Long-Only + Execution Smoothing v2 (n_steps=2)',
    'components': {
        'engines': ['FV3c', 'ML9'],
        'ensemble_weights': [0.6, 0.4],
        'position_type': 'long_only',
        'execution_smoothing': 'v2 (n_steps=2)',
        'regime_filter': False,
        'risk_overlay': False
    },
    'metrics': v1_4_metrics,
    'daily_returns': {
        'index': [d.strftime('%Y-%m-%d') for d in v1_4_returns.index],
        'values': v1_4_returns.tolist()
    } if len(v1_4_returns) > 0 else {'index': [], 'values': []},
    'rebalance_dates': [d.strftime('%Y-%m-%d') for d in rebalance_dates]
}

with open('results/v1_4_long_only_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to results/v1_4_long_only_results.json")

# Compare with v1.2
print("\n" + "="*100)
print("Comparison with v1.2")
print("="*100)

try:
    with open('results/v1_2_core_results.json', 'r') as f:
        v1_2_data = json.load(f)
    
    v1_2_sharpe = v1_2_data['metrics']['sharpe']
    v1_2_ann_ret = v1_2_data['metrics']['annual_return']
    
    print(f"\nv1.2 Baseline:")
    print(f"  Sharpe:       {v1_2_sharpe:.2f}")
    print(f"  Annual Ret:   {v1_2_ann_ret:.2%}")
    
    print(f"\nv1.4 (with Execution Smoothing v2):")
    print(f"  Sharpe:       {v1_4_metrics['sharpe']:.2f}")
    print(f"  Annual Ret:   {v1_4_metrics['annual_return']:.2%}")
    
    sharpe_improvement = v1_4_metrics['sharpe'] - v1_2_sharpe
    print(f"\nImprovement:")
    print(f"  Sharpe Δ:     {sharpe_improvement:+.2f}")
    
except Exception as e:
    print(f"\nCould not load v1.2 results: {e}")

print("\n" + "="*100)
print("v1.4 Complete Backtest Finished!")
print("="*100)
