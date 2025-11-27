#!/usr/bin/env python3
"""
v1.4 전체 백테스트 파이프라인
FV4 (Signal Smoothing) + ML10 (Signal Smoothing) + Execution Smoothing v2
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 모듈 경로 추가
sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.signal_prices import (
    SignalSmoothingConfig,
    build_signal_price_df,
    expand_signal_prices,
    get_rebalance_dates_from_signal_df
)
from utils.execution_smoothing_v2 import (
    ExecutionSmoothingConfig,
    portfolio_returns_with_execution_smoothing
)
from engines.factor_value_v4_signal_smoothing import FactorValueV4, FV4Config
from engines.ml_xgboost_v10_signal_smoothing import MLXGBoostV10, ML10Config
from utils.factors import compute_value_proxy, compute_momentum, compute_volatility

logger.info("="*100)
logger.info("v1.4 Full Pipeline Backtest")
logger.info("="*100)

# 1. 가격 데이터 로드
logger.info("\n[1/6] Loading price data...")
prices_df = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices_df = prices_df.sort_index()
logger.info(f"  Loaded {len(prices_df)} days, {len(prices_df.columns)} tickers")
logger.info(f"  Period: {prices_df.index[0]} ~ {prices_df.index[-1]}")

# 2. Signal Prices 생성
logger.info("\n[2/6] Building signal prices...")
cfg_signal = SignalSmoothingConfig(window=3)
signal_df_raw = build_signal_price_df(prices_df, cfg_signal)
signal_df = expand_signal_prices(signal_df_raw, prices_df.index)
rebalance_dates = get_rebalance_dates_from_signal_df(signal_df_raw)
logger.info(f"  Signal prices generated for {len(signal_df_raw)} rebalance dates")
logger.info(f"  First rebal: {rebalance_dates[0]}, Last rebal: {rebalance_dates[-1]}")

# 3. FV4 엔진 실행
logger.info("\n[3/6] Running FV4 engine...")
fv4_engine = FactorValueV4Engine(
    top_n=10,
    vol_window=30,
    mom_window=60
)
weights_fv4 = fv4_engine.generate_weights(
    prices=prices_df,
    signal_prices=signal_df,
    rebalance_dates=rebalance_dates
)
logger.info(f"  FV4 generated {len(weights_fv4)} weight sets")

# 4. ML10 엔진 실행
logger.info("\n[4/6] Running ML10 engine...")
ml10_engine = MLXGBoostV10Engine(
    top_n=10,
    lookback=252,
    horizon=21
)
weights_ml10 = ml10_engine.generate_weights(
    prices=prices_df,
    signal_prices=signal_df,
    rebalance_dates=rebalance_dates
)
logger.info(f"  ML10 generated {len(weights_ml10)} weight sets")

# 5. 앙상블 (60:40)
logger.info("\n[5/6] Creating ensemble...")
weights_ensemble = {}
for date in rebalance_dates:
    if date in weights_fv4 and date in weights_ml10:
        w_fv4 = weights_fv4[date]
        w_ml10 = weights_ml10[date]
        # 60:40 앙상블
        w_ens = 0.6 * w_fv4 + 0.4 * w_ml10
        # 정규화
        w_ens = w_ens / w_ens.sum()
        weights_ensemble[date] = w_ens

logger.info(f"  Ensemble created for {len(weights_ensemble)} dates")

# 6. Execution Smoothing v2 적용
logger.info("\n[6/6] Applying Execution Smoothing v2...")
cfg_exec = ExecutionSmoothingConfig(n_steps=2)
returns_v14 = portfolio_returns_with_execution_smoothing(
    prices=prices_df,
    weights_by_date=weights_ensemble,
    rebalance_dates=rebalance_dates,
    config=cfg_exec
)
logger.info(f"  Returns calculated for {len(returns_v14)} days")

# 성과 계산
def calculate_metrics(returns):
    total_ret = (1 + returns).prod() - 1
    n_days = len(returns)
    n_years = n_days / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
        'max_drawdown': max_dd,
        'total_return': total_ret,
        'win_rate': win_rate,
        'n_days': n_days
    }

metrics = calculate_metrics(returns_v14)

logger.info("\n" + "="*100)
logger.info("v1.4 Full Pipeline Performance")
logger.info("="*100)
logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
logger.info(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
logger.info(f"Total Return: {metrics['total_return']:.2%}")
logger.info(f"Days: {metrics['n_days']}")

# 결과 저장
results = {
    'version': 'v1.4_full',
    'description': 'FV4 + ML10 + Execution Smoothing v2',
    'config': {
        'signal_smoothing_window': cfg_signal.window,
        'execution_smoothing_steps': cfg_exec.n_steps,
        'fv4_top_n': fv4_engine.top_n,
        'ml10_top_n': ml10_engine.top_n,
        'ensemble_weights': {'fv4': 0.6, 'ml10': 0.4}
    },
    'metrics': metrics,
    'daily_returns': {
        'index': [d.strftime('%Y-%m-%d') for d in returns_v14.index],
        'values': returns_v14.tolist()
    }
}

with open('results/v1_4_full_results.json', 'w') as f:
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    json.dump(convert(results), f, indent=2)

logger.info("\n✅ Results saved to results/v1_4_full_results.json")

print("\n" + "="*100)
print("v1.4 Full Pipeline Backtest Complete!")
print("="*100)
print(f"\n🎯 Final Performance:")
print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  - Annual Return: {metrics['annual_return']:.2%}")
print(f"  - Annual Volatility: {metrics['annual_volatility']:.2%}")
print(f"  - Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"  - Win Rate: {metrics['win_rate']:.2%}")
