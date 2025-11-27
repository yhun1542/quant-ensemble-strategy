#!/usr/bin/env python3
"""
Factor Value v3c (Dynamic) - Volatility-based position sizing
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class FactorValueV3cDynamic:
    """
    Factor Value v3c - Dynamic position sizing
    - Single factor: value_proxy
    - Position size inversely proportional to volatility
    """
    
    def __init__(self, price_data: pd.DataFrame, factor_data: pd.DataFrame,
                 top_quantile: float = 0.2):
        self.prices = price_data
        self.factors = factor_data
        self.top_quantile = top_quantile
        
    def _calc_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """성과 지표 계산"""
        returns = returns.fillna(0.0)
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else 0.0
        annual_return = mean_ret * TRADING_DAYS
        annual_vol = std_ret * np.sqrt(TRADING_DAYS)
        
        cum_ret = (1.0 + returns).cumprod()
        peak = cum_ret.cummax()
        dd = cum_ret / peak - 1.0
        max_dd = dd.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return PerformanceMetrics(
            sharpe=float(sharpe),
            annual_return=float(annual_return),
            annual_volatility=float(annual_vol),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            num_trades=len(returns)
        )
    
    def _get_monthly_rebalance_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """월간 리밸런싱 날짜 생성"""
        dates = self.prices.loc[start:end].index
        monthly_dates = []
        
        current_month = None
        for date in dates:
            if current_month != date.month:
                monthly_dates.append(date)
                current_month = date.month
        
        return monthly_dates
    
    def _construct_portfolio(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        특정 날짜의 포트폴리오 구성
        - value_proxy 기준 선택
        - 포지션 크기는 변동성에 반비례
        """
        if date not in self.factors.index.get_level_values("date"):
            return {}
        
        factors_at_date = self.factors.loc[date].copy()
        factors_sorted = factors_at_date.sort_values("value_proxy", ascending=True)
        
        n_stocks = len(factors_sorted)
        n_long = int(n_stocks * self.top_quantile)
        n_short = int(n_stocks * self.top_quantile)
        
        long_tickers = factors_sorted.head(n_long).index.tolist()
        short_tickers = factors_sorted.tail(n_short).index.tolist()
        
        # 변동성 기반 가중치 계산
        portfolio = {}
        
        # Long positions
        long_vols = []
        for ticker in long_tickers:
            vol = factors_at_date.loc[ticker, "volatility_30d"]
            if vol > 0:
                long_vols.append((ticker, 1.0 / vol))
        
        if long_vols:
            total_inv_vol = sum(w for _, w in long_vols)
            for ticker, inv_vol in long_vols:
                portfolio[ticker] = inv_vol / total_inv_vol
        
        # Short positions
        short_vols = []
        for ticker in short_tickers:
            vol = factors_at_date.loc[ticker, "volatility_30d"]
            if vol > 0:
                short_vols.append((ticker, 1.0 / vol))
        
        if short_vols:
            total_inv_vol = sum(w for _, w in short_vols)
            for ticker, inv_vol in short_vols:
                portfolio[ticker] = -inv_vol / total_inv_vol
        
        return portfolio
    
    def _backtest_period(self, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.Series:
        """특정 기간 백테스트"""
        rebal_dates = self._get_monthly_rebalance_dates(test_start, test_end)
        
        daily_returns = []
        current_portfolio = {}
        
        test_dates = self.prices.loc[test_start:test_end].index
        
        for i, date in enumerate(test_dates):
            if i > 0 and date in rebal_dates:
                prev_date = test_dates[i-1]
                current_portfolio = self._construct_portfolio(prev_date)
            
            if current_portfolio and i > 0:
                prev_date = test_dates[i-1]
                
                daily_ret = 0.0
                for ticker, weight in current_portfolio.items():
                    if ticker in self.prices.columns:
                        ret = self.prices.loc[date, ticker] / self.prices.loc[prev_date, ticker] - 1.0
                        daily_ret += weight * ret
                
                daily_returns.append({"date": date, "ret": daily_ret})
        
        if daily_returns:
            return pd.Series({r["date"]: r["ret"] for r in daily_returns})
        else:
            return pd.Series(dtype=float)
    
    def run_walkforward_backtest(self) -> Dict[str, Any]:
        """Walk-forward backtest"""
        dates = sorted(set(self.factors.index.get_level_values("date")))
        
        train_years = 3
        test_years = 1
        
        start_year = dates[0].year
        end_year = dates[-1].year - test_years
        
        windows = []
        for y in range(start_year + train_years, end_year + 1):
            test_start = pd.Timestamp(year=y, month=1, day=1)
            test_end = pd.Timestamp(year=y+test_years-1, month=12, day=31)
            windows.append((test_start, test_end))
        
        all_daily_ret = []
        
        for (te_start, te_end) in windows:
            daily_ret = self._backtest_period(te_start, te_end)
            
            if len(daily_ret) > 0:
                all_daily_ret.append(daily_ret)
        
        if not all_daily_ret:
            raise RuntimeError("No valid walk-forward windows.")
        
        daily_ret_all = pd.concat(all_daily_ret).sort_index()
        overall_metrics = self._calc_metrics(daily_ret_all)
        
        return {
            "overall": asdict(overall_metrics),
            "daily_returns": [
                {"date": d.strftime("%Y-%m-%d"), "ret": float(r)}
                for d, r in daily_ret_all.items()
            ],
        }


def main():
    print("=" * 100)
    print("Factor Value v3c (Dynamic Position Sizing)")
    print("=" * 100)
    
    # 데이터 로드
    price_data = pd.read_parquet("data/price_data_sp500.parquet")
    factor_data = pd.read_parquet("data/factors_price_based.parquet")
    
    engine = FactorValueV3cDynamic(price_data, factor_data, top_quantile=0.2)
    result = engine.run_walkforward_backtest()
    
    # 저장
    output_path = Path("engine_results/factor_value_v3c_dynamic_oos.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 100)
    print("Overall Performance (Out-of-Sample)")
    print("=" * 100)
    print(f"Sharpe Ratio: {result['overall']['sharpe']:.4f}")
    print(f"Annual Return: {result['overall']['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {result['overall']['annual_volatility']*100:.2f}%")
    print(f"Max Drawdown: {result['overall']['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {result['overall']['win_rate']*100:.2f}%")
    
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
