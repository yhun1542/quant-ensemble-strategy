#!/usr/bin/env python3
"""
Generate Weights for v1.4

Run FV3c and ML9 engines to generate weights_by_date for each rebalance date.
This is the FULL ENGINE-LEVEL implementation required for v1.4.
"""
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.factors import compute_all_factors

print("="*100)
print("Generating Weights for v1.4 - FULL ENGINE-LEVEL Implementation")
print("="*100)

# Load price data
print("\n[1/6] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Compute factors
print("\n[2/6] Computing factors...")
factors = compute_all_factors(prices)
print(f"  Computed factors for {len(factors)} data points")
print(f"  Factor columns: {factors.columns.tolist()}")

# Get monthly rebalance dates
print("\n[3/6] Determining rebalance dates...")
dates = prices.index
monthly_dates = []
current_month = None
for date in dates:
    if current_month != date.month:
        monthly_dates.append(date)
        current_month = date.month

rebalance_dates = monthly_dates[1:]  # Skip first month (no prior data)
print(f"  Generated {len(rebalance_dates)} rebalance dates")

# FV3c Engine: Generate weights
print("\n[4/6] Running FV3c engine to generate weights...")
fv3c_weights_by_date = {}
top_quantile = 0.2

for i, rebal_date in enumerate(rebalance_dates):
    if rebal_date not in factors.index.get_level_values("date"):
        continue
    
    # Get factors at rebalance date
    factors_at_date = factors.loc[rebal_date].copy()
    
    # Sort by value_proxy (ascending = cheap first)
    factors_sorted = factors_at_date.sort_values("value_proxy", ascending=True)
    
    n_stocks = len(factors_sorted)
    n_long = int(n_stocks * top_quantile)
    n_short = int(n_stocks * top_quantile)
    
    # Select long/short tickers
    long_tickers = factors_sorted.head(n_long).index.tolist()
    short_tickers = factors_sorted.tail(n_short).index.tolist()
    
    # Calculate volatility-based weights
    portfolio = {}
    
    # Long positions (inverse volatility weighting)
    long_vols = []
    for ticker in long_tickers:
        vol = factors_at_date.loc[ticker, "volatility_30d"]
        if vol > 0:
            long_vols.append((ticker, 1.0 / vol))
    
    if long_vols:
        total_inv_vol = sum(w for _, w in long_vols)
        for ticker, inv_vol in long_vols:
            portfolio[ticker] = inv_vol / total_inv_vol
    
    # Short positions (inverse volatility weighting)
    short_vols = []
    for ticker in short_tickers:
        vol = factors_at_date.loc[ticker, "volatility_30d"]
        if vol > 0:
            short_vols.append((ticker, 1.0 / vol))
    
    if short_vols:
        total_inv_vol = sum(w for _, w in short_vols)
        for ticker, inv_vol in short_vols:
            portfolio[ticker] = -inv_vol / total_inv_vol
    
    fv3c_weights_by_date[rebal_date] = pd.Series(portfolio)
    
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/{len(rebalance_dates)} dates...")

print(f"  ✅ FV3c weights generated for {len(fv3c_weights_by_date)} dates")

# ML9 Engine: Generate weights
print("\n[5/6] Running ML9 engine to generate weights...")
import xgboost as xgb

ml9_weights_by_date = {}
prediction_horizon = 10
train_window = 252 * 2  # 2 years

# Prepare features
feature_cols = ["momentum_60d", "volatility_30d", "value_proxy"]

for i, rebal_date in enumerate(rebalance_dates):
    if rebal_date not in factors.index.get_level_values("date"):
        continue
    
    # Get training data (past 2 years)
    train_start_idx = max(0, dates.get_loc(rebal_date) - train_window)
    train_dates = dates[train_start_idx:dates.get_loc(rebal_date)]
    
    if len(train_dates) < 60:  # Need at least 60 days
        continue
    
    # Build training dataset
    X_train_list = []
    y_train_list = []
    
    for train_date in train_dates:
        if train_date not in factors.index.get_level_values("date"):
            continue
        
        factors_at_train = factors.loc[train_date]
        
        # Calculate forward returns
        date_idx = dates.get_loc(train_date)
        if date_idx + prediction_horizon >= len(dates):
            continue
        
        future_date = dates[date_idx + prediction_horizon]
        
        for ticker in factors_at_train.index:
            if ticker not in prices.columns:
                continue
            
            # Features
            features = factors_at_train.loc[ticker, feature_cols].values
            
            # Target: forward return
            try:
                fwd_ret = prices.loc[future_date, ticker] / prices.loc[train_date, ticker] - 1.0
            except:
                continue
            
            X_train_list.append(features)
            y_train_list.append(fwd_ret)
    
    if len(X_train_list) < 100:  # Need enough training samples
        continue
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    # Convert to quantile-based classification
    # Top 20% = class 2, Bottom 20% = class 0, Middle = class 1
    y_quantiles = pd.Series(y_train).rank(pct=True)
    y_class = np.ones(len(y_train), dtype=int)  # Default: middle
    y_class[y_quantiles >= 0.8] = 2  # Top 20%
    y_class[y_quantiles <= 0.2] = 0  # Bottom 20%
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_class)
    
    # Predict on current date
    factors_at_rebal = factors.loc[rebal_date]
    X_pred = factors_at_rebal[feature_cols].values
    
    # Get probabilities for each class
    probs = model.predict_proba(X_pred)
    
    # Calculate score: P(class=2) - P(class=0)
    scores = probs[:, 2] - probs[:, 0]
    scores_series = pd.Series(scores, index=factors_at_rebal.index)
    
    # Select top/bottom by score
    scores_sorted = scores_series.sort_values(ascending=False)
    
    n_stocks = len(scores_sorted)
    n_long = int(n_stocks * top_quantile)
    n_short = int(n_stocks * top_quantile)
    
    long_tickers = scores_sorted.head(n_long).index.tolist()
    short_tickers = scores_sorted.tail(n_short).index.tolist()
    
    # Equal weighting for ML9
    portfolio = {}
    
    if long_tickers:
        for ticker in long_tickers:
            portfolio[ticker] = 1.0 / len(long_tickers)
    
    if short_tickers:
        for ticker in short_tickers:
            portfolio[ticker] = -1.0 / len(short_tickers)
    
    ml9_weights_by_date[rebal_date] = pd.Series(portfolio)
    
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/{len(rebalance_dates)} dates...")

print(f"  ✅ ML9 weights generated for {len(ml9_weights_by_date)} dates")

# Save results
print("\n[6/6] Saving results...")

# Convert to JSON format
fv3c_weights_json = {}
for date, weights in fv3c_weights_by_date.items():
    date_str = date.strftime('%Y-%m-%d')
    fv3c_weights_json[date_str] = weights.to_dict()

ml9_weights_json = {}
for date, weights in ml9_weights_by_date.items():
    date_str = date.strftime('%Y-%m-%d')
    ml9_weights_json[date_str] = weights.to_dict()

output = {
    'generated_at': datetime.now().isoformat(),
    'description': 'FV3c and ML9 engine weights for v1.4 implementation',
    'config': {
        'top_quantile': top_quantile,
        'prediction_horizon': prediction_horizon,
        'train_window_days': train_window
    },
    'fv3c_weights': fv3c_weights_json,
    'ml9_weights': ml9_weights_json,
    'rebalance_dates': sorted(set(fv3c_weights_json.keys()) & set(ml9_weights_json.keys()))
}

with open('results/ensemble_fv3c_ml9.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"  ✅ Saved to results/ensemble_fv3c_ml9.json")
print(f"  Total rebalance dates: {len(output['rebalance_dates'])}")

print("\n" + "="*100)
print("Weights Generation Complete!")
print("="*100)
print("\nNext step: Run backtest_v1_4_complete.py to apply Execution Smoothing v2")
