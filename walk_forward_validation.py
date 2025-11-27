#!/usr/bin/env python3
"""
Walk-Forward Validation for v1.5

This script performs walk-forward analysis to assess overfitting risk more accurately.
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

print("="*100)
print("Walk-Forward Validation for v1.5")
print("="*100)

# Load v1.5 results
print("\n[1/4] Loading v1.5 results...")
with open('results/v1_5_with_costs_results.json', 'r') as f:
    v1_5_data = json.load(f)

net_returns_dict = v1_5_data['daily_returns']['net']
net_returns = pd.Series(
    net_returns_dict['values'],
    index=pd.to_datetime(net_returns_dict['index'])
)

print(f"  Loaded {len(net_returns)} days of net returns")
print(f"  Date range: {net_returns.index[0].date()} to {net_returns.index[-1].date()}")

# Define walk-forward windows
print("\n[2/4] Defining walk-forward windows...")

# Use 6-month windows with 3-month step
window_months = 12  # 12 months per window
step_months = 6     # 6 months step

all_dates = net_returns.index
start_date = all_dates[0]
end_date = all_dates[-1]

windows = []
current_start = start_date

while current_start < end_date:
    # Window end is 12 months after start
    window_end = current_start + pd.DateOffset(months=window_months)
    
    if window_end > end_date:
        window_end = end_date
    
    # Get returns in this window
    window_returns = net_returns[(net_returns.index >= current_start) & 
                                  (net_returns.index < window_end)]
    
    if len(window_returns) > 30:  # At least 30 days
        windows.append({
            'start': current_start,
            'end': window_end,
            'returns': window_returns
        })
    
    # Move to next window
    current_start = current_start + pd.DateOffset(months=step_months)

print(f"  Created {len(windows)} walk-forward windows")

# Calculate metrics for each window
print("\n[3/4] Calculating metrics for each window...")

def calc_metrics(rets):
    if len(rets) == 0:
        return {}
    
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1 if n_years > 0 and total > -1 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 and not np.isnan(ann_ret) else np.nan
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    win_rate = (rets > 0).sum() / len(rets)
    
    return {
        'sharpe': sharpe,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'total_ret': total,
        'win_rate': win_rate,
        'n_days': len(rets)
    }

window_metrics = []
for i, window in enumerate(windows):
    metrics = calc_metrics(window['returns'])
    metrics['start'] = window['start']
    metrics['end'] = window['end']
    window_metrics.append(metrics)
    
    print(f"  Window {i+1}: {window['start'].date()} to {window['end'].date()}")
    print(f"    Sharpe: {metrics['sharpe']:.2f}, Ann Ret: {metrics['ann_ret']:.2%}, "
          f"Max DD: {metrics['max_dd']:.2%}, Days: {metrics['n_days']}")

# Analyze consistency
print("\n[4/4] Analyzing performance consistency...")

sharpes = [m['sharpe'] for m in window_metrics if not np.isnan(m['sharpe'])]
ann_rets = [m['ann_ret'] for m in window_metrics if not np.isnan(m['ann_ret'])]

if sharpes:
    sharpe_mean = np.mean(sharpes)
    sharpe_std = np.std(sharpes)
    sharpe_min = np.min(sharpes)
    sharpe_max = np.max(sharpes)
    
    print(f"\nSharpe Ratio across windows:")
    print(f"  Mean:   {sharpe_mean:.2f}")
    print(f"  Std:    {sharpe_std:.2f}")
    print(f"  Min:    {sharpe_min:.2f}")
    print(f"  Max:    {sharpe_max:.2f}")
    print(f"  Range:  {sharpe_max - sharpe_min:.2f}")

if ann_rets:
    ret_mean = np.mean(ann_rets)
    ret_std = np.std(ann_rets)
    ret_min = np.min(ann_rets)
    ret_max = np.max(ann_rets)
    
    print(f"\nAnnual Return across windows:")
    print(f"  Mean:   {ret_mean:.2%}")
    print(f"  Std:    {ret_std:.2%}")
    print(f"  Min:    {ret_min:.2%}")
    print(f"  Max:    {ret_max:.2%}")

# Compare with full-period performance
full_metrics = calc_metrics(net_returns)

print(f"\nFull-Period Performance:")
print(f"  Sharpe: {full_metrics['sharpe']:.2f}")
print(f"  Ann Ret: {full_metrics['ann_ret']:.2%}")

# Consistency score
if sharpes:
    consistency_score = 1.0 - (sharpe_std / abs(sharpe_mean)) if sharpe_mean != 0 else 0
    print(f"\nConsistency Score: {consistency_score:.2f}")
    print(f"  (1.0 = perfect consistency, 0.0 = high variability)")

# Assessment
print("\n" + "="*100)
print("Walk-Forward Validation Assessment")
print("="*100)

if sharpe_std / abs(sharpe_mean) < 0.3:
    print("\n✅ GOOD CONSISTENCY")
    print("  Strategy shows consistent performance across different time windows.")
    print("  Overfitting risk is LOW.")
elif sharpe_std / abs(sharpe_mean) < 0.5:
    print("\n⚠️ MODERATE CONSISTENCY")
    print("  Strategy shows some variability across time windows.")
    print("  Overfitting risk is MODERATE.")
else:
    print("\n❌ POOR CONSISTENCY")
    print("  Strategy shows high variability across time windows.")
    print("  Overfitting risk is HIGH.")

# Save results
wf_results = {
    'strategy': 'v1.5_walk_forward',
    'window_months': window_months,
    'step_months': step_months,
    'n_windows': len(windows),
    'window_metrics': [
        {
            'start': m['start'].strftime('%Y-%m-%d'),
            'end': m['end'].strftime('%Y-%m-%d'),
            'sharpe': m['sharpe'],
            'ann_ret': m['ann_ret'],
            'ann_vol': m['ann_vol'],
            'max_dd': m['max_dd'],
            'n_days': m['n_days']
        }
        for m in window_metrics
    ],
    'summary': {
        'sharpe_mean': sharpe_mean,
        'sharpe_std': sharpe_std,
        'sharpe_min': sharpe_min,
        'sharpe_max': sharpe_max,
        'ret_mean': ret_mean,
        'ret_std': ret_std,
        'consistency_score': consistency_score
    },
    'full_period': full_metrics
}

with open('results/v1_5_walk_forward_results.json', 'w') as f:
    json.dump(wf_results, f, indent=2)

print("\n✅ Results saved to results/v1_5_walk_forward_results.json")

print("\n" + "="*100)
print("Walk-Forward Validation Complete")
print("="*100)
