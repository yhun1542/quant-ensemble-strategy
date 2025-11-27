"""
QV v2.1 Backtest with Inverse-Vol Weighting

Goal: Achieve Sharpe 1.5-2.0 by reducing volatility through:
1. Inverse-vol weighting (low vol stocks get higher weight)
2. Expanded top quantile (30% instead of 20%)
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
import json
from datetime import datetime

from data_loader_sf1 import SF1Config, load_sf1_raw, expand_sf1_to_daily
from engines.factor_quality_value_v2_1 import FactorQVEngineV21
from utils.transaction_costs import TransactionCostModel

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
    print("QV v2.1 ENGINE - INVERSE-VOL WEIGHTING")
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
    
    # 4. Run QV v2.1 engine - Multiple configurations
    print("\n" + "="*100)
    print("RUNNING QV v2.1 ENGINE - TESTING CONFIGURATIONS")
    print("="*100)
    
    configs = [
        # (top_quantile, use_inverse_vol, name)
        (0.2, False, "QV_v2.0_EqualWeight_Top20"),
        (0.2, True,  "QV_v2.1_InverseVol_Top20"),
        (0.3, False, "QV_v2.1_EqualWeight_Top30"),
        (0.3, True,  "QV_v2.1_InverseVol_Top30"),
        (0.4, True,  "QV_v2.1_InverseVol_Top40"),
    ]
    
    all_results = []
    
    for top_q, use_inv_vol, name in configs:
        print(f"\n--- {name} ---")
        print(f"Top Quantile: {top_q:.1%}, Inverse-Vol: {use_inv_vol}")
        
        engine = FactorQVEngineV21(
            top_quantile=top_q,
            long_gross=1.0,
            short_gross=0.0,
            long_only=True,
            use_inverse_vol=use_inv_vol,
            vol_lookback=63,
        )
        
        weights_by_date = engine.build_portfolio(
            fundamentals_daily,
            prices,
            rebalance_dates,
        )
        
        print(f"Generated weights for {len(weights_by_date)} rebalance dates")
        
        # Calculate returns (gross)
        returns_gross = portfolio_returns(prices, weights_by_date)
        print(f"Portfolio returns: {len(returns_gross)} days")
        
        # Apply transaction costs
        tc_model = TransactionCostModel(
            commission_bps=0.5,
            spread_bps=5.0,
            impact_bps=3.0,
        )
        
        returns_net = tc_model.apply_costs_to_returns(
            returns_gross,
            weights_by_date,
            list(weights_by_date.keys()),
        )
        
        tc_stats = {"total_bps": tc_model.total_bps}
        
        # Calculate metrics
        metrics_gross = calculate_metrics(returns_gross, f"{name}_Gross")
        metrics_net = calculate_metrics(returns_net, f"{name}_Net")
        
        print(f"\nGross Sharpe: {metrics_gross['sharpe_ratio']:.2f}")
        print(f"Net Sharpe:   {metrics_net['sharpe_ratio']:.2f}")
        print(f"Annual Vol:   {metrics_net['annual_vol']:.2%}")
        print(f"Annual Return: {metrics_net['annual_return']:.2%}")
        
        all_results.append({
            "config": {
                "name": name,
                "top_quantile": top_q,
                "use_inverse_vol": use_inv_vol,
            },
            "metrics_gross": metrics_gross,
            "metrics_net": metrics_net,
            "tc_stats": tc_stats,
        })
    
    # 5. Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    results = {
        "strategy": "QV_v2_1_Inverse_Vol",
        "date": datetime.now().isoformat(),
        "configurations": all_results,
    }
    
    with open("/home/ubuntu/quant-ensemble-strategy/results/v2_1_qv_inverse_vol_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: results/v2_1_qv_inverse_vol_results.json")
    
    # 6. Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    print(f"\n{'Config':<35s} {'Sharpe':<10s} {'Annual Ret':<12s} {'Annual Vol':<12s} {'Max DD':<10s}")
    print("-" * 85)
    
    for res in all_results:
        name = res["config"]["name"]
        m = res["metrics_net"]
        print(f"{name:<35s} {m['sharpe_ratio']:>9.2f} {m['annual_return']:>11.2%} {m['annual_vol']:>11.2%} {m['max_drawdown']:>9.2%}")
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
