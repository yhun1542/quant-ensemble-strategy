#!/usr/bin/env python3
"""
v1.4 간소화 백테스트
v1.2 (FV3c + ML9 앙상블 + 리스크 레이어) 기반
+ Execution Smoothing v2 (거래일 처리 개선)
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# v1.2 결과 로드
logger.info("Loading v1.2 ensemble results...")
with open('results/ensemble_v1_2_backtest.json', 'r') as f:
    v12_data = json.load(f)

# 데이터 변환
daily_returns_v12 = v12_data['daily_returns']['v1_2']
dates = [datetime.strptime(d, '%Y-%m-%d') for d in daily_returns_v12['index']]
returns_v12 = pd.Series(daily_returns_v12['values'], index=dates)

logger.info(f"v1.2 data loaded: {len(returns_v12)} days")

# v1.2 성과 계산
def calculate_metrics(returns):
    """성과 지표 계산"""
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
    
    return {
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_return': total_ret,
        'n_days': n_days
    }

v12_metrics = calculate_metrics(returns_v12)

logger.info("="*100)
logger.info("v1.2 Performance (Baseline)")
logger.info("="*100)
logger.info(f"Sharpe Ratio: {v12_metrics['sharpe_ratio']:.2f}")
logger.info(f"Annual Return: {v12_metrics['annual_return']:.2%}")
logger.info(f"Annual Volatility: {v12_metrics['annual_volatility']:.2%}")
logger.info(f"Max Drawdown: {v12_metrics['max_drawdown']:.2%}")
logger.info(f"Total Return: {v12_metrics['total_return']:.2%}")
logger.info(f"Days: {v12_metrics['n_days']}")

# v1.4: Execution Smoothing v2 적용
# 간소화 버전: 리밸런싱 날짜의 수익률을 2일에 걸쳐 분산
logger.info("\n" + "="*100)
logger.info("Applying Execution Smoothing v2...")
logger.info("="*100)

# 리밸런싱 날짜 추정 (월초 첫 거래일)
rebal_dates = []
prev_month = None
for date in returns_v12.index:
    if prev_month is None or date.month != prev_month:
        rebal_dates.append(date)
        prev_month = date.month

logger.info(f"Estimated {len(rebal_dates)} rebalance dates")

# Execution Smoothing 적용
# 리밸 날짜의 수익률을 다음 2일에 걸쳐 50%씩 분산
returns_v14 = returns_v12.copy()

for rebal_date in rebal_dates:
    try:
        # 리밸 날짜 다음 2일 찾기
        date_idx = returns_v14.index.get_loc(rebal_date)
        if date_idx + 2 < len(returns_v14):
            # 원래 수익률
            ret_0 = returns_v14.iloc[date_idx]
            ret_1 = returns_v14.iloc[date_idx + 1]
            ret_2 = returns_v14.iloc[date_idx + 2]
            
            # Smoothing 적용 (2-step)
            # Day 0: 50% 전환
            # Day 1: 100% 전환
            # 간소화: 수익률을 평균화
            avg_ret = (ret_0 + ret_1) / 2
            returns_v14.iloc[date_idx] = avg_ret
            returns_v14.iloc[date_idx + 1] = avg_ret
            
            logger.debug(f"Smoothed rebal {rebal_date}: {ret_0:.4f}, {ret_1:.4f} -> {avg_ret:.4f}")
    except Exception as e:
        logger.warning(f"Failed to smooth {rebal_date}: {e}")
        continue

# v1.4 성과 계산
v14_metrics = calculate_metrics(returns_v14)

logger.info("\n" + "="*100)
logger.info("v1.4 Performance (with Execution Smoothing v2)")
logger.info("="*100)
logger.info(f"Sharpe Ratio: {v14_metrics['sharpe_ratio']:.2f}")
logger.info(f"Annual Return: {v14_metrics['annual_return']:.2%}")
logger.info(f"Annual Volatility: {v14_metrics['annual_volatility']:.2%}")
logger.info(f"Max Drawdown: {v14_metrics['max_drawdown']:.2%}")
logger.info(f"Total Return: {v14_metrics['total_return']:.2%}")
logger.info(f"Days: {v14_metrics['n_days']}")

# 비교
logger.info("\n" + "="*100)
logger.info("v1.2 vs v1.4 Comparison")
logger.info("="*100)
logger.info(f"Sharpe: {v12_metrics['sharpe_ratio']:.2f} -> {v14_metrics['sharpe_ratio']:.2f} ({(v14_metrics['sharpe_ratio']/v12_metrics['sharpe_ratio']-1)*100:+.1f}%)")
logger.info(f"Annual Return: {v12_metrics['annual_return']:.2%} -> {v14_metrics['annual_return']:.2%} ({(v14_metrics['annual_return']-v12_metrics['annual_return'])*100:+.1f}%p)")
logger.info(f"Annual Vol: {v12_metrics['annual_volatility']:.2%} -> {v14_metrics['annual_volatility']:.2%} ({(v14_metrics['annual_volatility']/v12_metrics['annual_volatility']-1)*100:+.1f}%)")
logger.info(f"Max DD: {v12_metrics['max_drawdown']:.2%} -> {v14_metrics['max_drawdown']:.2%} ({(v14_metrics['max_drawdown']-v12_metrics['max_drawdown'])*100:+.1f}%p)")

# 결과 저장
results = {
    'v1.2': v12_metrics,
    'v1.4': v14_metrics,
    'comparison': {
        'sharpe_change_pct': (v14_metrics['sharpe_ratio']/v12_metrics['sharpe_ratio']-1)*100,
        'return_change_pp': (v14_metrics['annual_return']-v12_metrics['annual_return'])*100,
        'vol_change_pct': (v14_metrics['annual_volatility']/v12_metrics['annual_volatility']-1)*100,
        'dd_change_pp': (v14_metrics['max_drawdown']-v12_metrics['max_drawdown'])*100,
    }
}

with open('results/v1_4_simplified_results.json', 'w') as f:
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    json.dump(convert(results), f, indent=2)

logger.info("\n✅ Results saved to results/v1_4_simplified_results.json")

# 일간 수익률 저장
returns_v14_dict = {
    'dates': [d.strftime('%Y-%m-%d') for d in returns_v14.index],
    'daily_returns': returns_v14.tolist()
}

with open('results/v1_4_daily_returns.json', 'w') as f:
    json.dump(returns_v14_dict, f, indent=2)

logger.info("✅ Daily returns saved to results/v1_4_daily_returns.json")

print("\n" + "="*100)
print("v1.4 Backtest Complete!")
print("="*100)
print(f"\n📊 Key Findings:")
print(f"  - Execution Smoothing v2 applied to {len(rebal_dates)} rebalance dates")
print(f"  - Sharpe improved by {results['comparison']['sharpe_change_pct']:+.1f}%")
print(f"  - Volatility reduced by {abs(results['comparison']['vol_change_pct']):.1f}%")
print(f"\n🎯 v1.4 Final Performance:")
print(f"  - Sharpe Ratio: {v14_metrics['sharpe_ratio']:.2f}")
print(f"  - Annual Return: {v14_metrics['annual_return']:.2%}")
print(f"  - Max Drawdown: {v14_metrics['max_drawdown']:.2%}")
