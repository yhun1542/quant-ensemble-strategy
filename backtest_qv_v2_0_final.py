"""
v2.0 Quality-Value Engine - Final Backtest

Clean implementation with accurate metrics calculation.
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
import json
from datetime import datetime

from data_loader_sf1 import SF1Config, load_sf1_raw, expand_sf1_to_daily
from engines.factor_quality_value_v1 import FactorQVEngineV1

# API Key
SF1_API_KEY = "H6zH4Q2CDr9uTFk9koqJ"

# 30 tickers
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# SF1 indicators
INDICATORS_QV = [
    "pe", "pb", "ps", "evebitda",
    "roe", "ebitdamargin", "de", "currentratio",
]


def load_price_data():
    """Load price data"""
    df = pd.read_csv("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp500.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    available_tickers = [t for t in TICKERS if t in df.columns]
    df = df[available_tickers]
    return df


def calculate_metrics(returns: pd.Series, name: str = "Strategy") -> dict:
    """Calculate performance metrics"""
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


def portfolio_returns(
    prices: pd.DataFrame,
    weights_by_date: dict,
) -> pd.Series:
    """Calculate portfolio returns"""
    returns_daily = prices.pct_change()
    portfolio_returns = []
    
    current_weights = None
    
    for date in prices.index:
        # Check if rebalance
        if date in weights_by_date:
            current_weights = weights_by_date[date]
        
        if current_weights is None:
            continue
        
        # Calculate return
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


def main():
    print("="*100)
    print("v2.0 QV ENGINE - FINAL BACKTEST")
    print("="*100)
    
    # 1. Load price data
    print("\nLoading price data...")
    prices = load_price_data()
    print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # 2. Load SF1 fundamental data
    print("\n" + "="*100)
    print("LOADING SF1 FUNDAMENTAL DATA")
    print("="*100)
    
    cfg = SF1Config(
        api_key=SF1_API_KEY,
        dimension="ART",
        min_date="2020-01-01",
    )
    
    sf1_raw = load_sf1_raw(TICKERS, cfg, INDICATORS_QV)
    fundamentals_daily = expand_sf1_to_daily(
        sf1_raw,
        prices.index,
        shift_one_day=True,
    )
    
    print(f"Fundamentals daily shape: {fundamentals_daily.shape}")
    
    # 3. Generate rebalance dates
    print("\n" + "="*100)
    print("GENERATING REBALANCE DATES")
    print("="*100)
    
    rebalance_dates = []
    for year in range(prices.index.year.min(), prices.index.year.max() + 1):
        for month in range(1, 13):
            month_dates = prices.index[(prices.index.year == year) & (prices.index.month == month)]
            if len(month_dates) > 0:
                rebalance_dates.append(month_dates[0])
    
    rebalance_dates = sorted(rebalance_dates)
    print(f"Generated {len(rebalance_dates)} rebalance dates")
    
    # 4. Run QV engine - Long-Only
    print("\n" + "="*100)
    print("RUNNING QV ENGINE - LONG-ONLY")
    print("="*100)
    
    engine = FactorQVEngineV1(
        top_quantile=0.2,
        long_gross=1.0,
        short_gross=0.0,
        long_only=True,
    )
    
    weights_by_date = engine.build_portfolio(
        fundamentals_daily,
        rebalance_dates,
    )
    
    print(f"Generated weights for {len(weights_by_date)} rebalance dates")
    
    # Calculate returns
    returns = portfolio_returns(prices, weights_by_date)
    print(f"Portfolio returns: {len(returns)} days")
    
    # Calculate metrics
    metrics = calculate_metrics(returns, "QV Long-Only v2.0")
    
    print("\n" + "="*100)
    print("QV LONG-ONLY RESULTS")
    print("="*100)
    for k, v in metrics.items():
        if isinstance(v, float):
            if 'return' in k or 'drawdown' in k or 'rate' in k:
                print(f"{k:20s}: {v:8.2%}")
            else:
                print(f"{k:20s}: {v:8.2f}")
        else:
            print(f"{k:20s}: {v}")
    
    # Save results
    results = {
        "strategy": "QV_Long_Only_v2_0_Final",
        "date": datetime.now().isoformat(),
        "metrics": metrics,
        "rebalance_count": len(weights_by_date),
    }
    
    with open("/home/ubuntu/quant-ensemble-strategy/results/v2_0_qv_final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: results/v2_0_qv_final_results.json")
    
    # 5. Compare with v1.5
    print("\n" + "="*100)
    print("COMPARISON WITH v1.5")
    print("="*100)
    
    try:
        with open("/home/ubuntu/quant-ensemble-strategy/results/v1_5_with_costs_results.json") as f:
            v15_results = json.load(f)
        
        v15_metrics = v15_results.get("metrics_net", {})
        
        print(f"\n{'Metric':<20s} {'v1.5 (Net)':<15s} {'QV v2.0':<15s} {'Diff':<15s}")
        print("-" * 70)
        
        compare_keys = ["sharpe_ratio", "annual_return", "annual_vol", "max_drawdown", "win_rate"]
        for key in compare_keys:
            v15_val = v15_metrics.get(key, 0)
            qv_val = metrics.get(key, 0)
            diff = qv_val - v15_val
            
            if 'return' in key or 'drawdown' in key or 'rate' in key:
                print(f"{key:<20s} {v15_val:>14.2%} {qv_val:>14.2%} {diff:>+14.2%}")
            else:
                print(f"{key:<20s} {v15_val:>14.2f} {qv_val:>14.2f} {diff:>+14.2f}")
        
    except FileNotFoundError:
        print("v1.5 results not found, skipping comparison")
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
