#!/usr/bin/env python3
"""
Factor Value v4 - Signal Smoothing 지원
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd
import numpy as np


@dataclass
class FV4Config:
    """FV4 설정"""
    top_quantile: float = 0.2  # 상위/하위 20% 선택
    

class FactorValueV4:
    """
    Factor Value v4 - Signal Smoothing 지원
    
    - Signal prices를 사용하여 팩터 계산
    - Value proxy 기준 종목 선택
    - 변동성 역가중 포지션 사이징
    """
    
    def __init__(self, cfg: FV4Config | None = None):
        self.cfg = cfg or FV4Config()
    
    def build_portfolio(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        포트폴리오 구성
        
        Parameters
        ----------
        prices : pd.DataFrame
            실제 가격 (index=date, columns=tickers)
        factors : pd.DataFrame
            팩터 데이터 (MultiIndex: date, ticker)
        rebalance_dates : list
            리밸런싱 날짜 리스트
        
        Returns
        -------
        dict
            {date: Series(ticker -> weight)}
        """
        weights_by_date = {}
        
        for d in rebalance_dates:
            if d not in factors.index.get_level_values("date"):
                continue
            
            factors_at_date = factors.loc[d].copy()
            
            # value_proxy 기준 정렬 (낮을수록 저평가)
            factors_sorted = factors_at_date.sort_values("value_proxy", ascending=True)
            
            n_stocks = len(factors_sorted)
            n_long = int(n_stocks * self.cfg.top_quantile)
            n_short = int(n_stocks * self.cfg.top_quantile)
            
            long_tickers = factors_sorted.head(n_long).index.tolist()
            short_tickers = factors_sorted.tail(n_short).index.tolist()
            
            # 변동성 역가중
            portfolio = {}
            
            # Long positions
            long_vols = []
            for ticker in long_tickers:
                vol = factors_at_date.loc[ticker, "volatility_30d"]
                if vol > 0 and not np.isnan(vol):
                    long_vols.append((ticker, 1.0 / vol))
            
            if long_vols:
                total_inv_vol = sum(w for _, w in long_vols)
                for ticker, inv_vol in long_vols:
                    portfolio[ticker] = inv_vol / total_inv_vol
            
            # Short positions
            short_vols = []
            for ticker in short_tickers:
                vol = factors_at_date.loc[ticker, "volatility_30d"]
                if vol > 0 and not np.isnan(vol):
                    short_vols.append((ticker, 1.0 / vol))
            
            if short_vols:
                total_inv_vol = sum(w for _, w in short_vols)
                for ticker, inv_vol in short_vols:
                    portfolio[ticker] = -inv_vol / total_inv_vol
            
            if portfolio:
                weights_by_date[d] = pd.Series(portfolio)
        
        return weights_by_date


def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: list[pd.Timestamp],
) -> pd.Series:
    """
    포트폴리오 일간 수익률 계산
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터
    weights_by_date : dict
        {date: Series(ticker -> weight)}
    rebalance_dates : list
        리밸런싱 날짜
    
    Returns
    -------
    pd.Series
        일간 수익률
    """
    daily_returns = []
    current_weights = None
    
    for i in range(len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        
        # 리밸런싱 체크
        if date in rebalance_dates and date in weights_by_date:
            current_weights = weights_by_date[date]
        
        # 수익률 계산
        if current_weights is not None and len(current_weights) > 0:
            daily_ret = 0.0
            for ticker in current_weights.index:
                if ticker in prices.columns:
                    ret = prices.loc[next_date, ticker] / prices.loc[date, ticker] - 1.0
                    daily_ret += current_weights[ticker] * ret
            
            daily_returns.append({"date": next_date, "ret": daily_ret})
    
    if daily_returns:
        return pd.Series({r["date"]: r["ret"] for r in daily_returns})
    else:
        return pd.Series(dtype=float)


if __name__ == "__main__":
    print("="*100)
    print("Factor Value v4 - Signal Smoothing 지원")
    print("="*100)
    print("\n사용 예시:")
    print("  from engines.factor_value_v4_signal_smoothing import FactorValueV4, FV4Config")
    print("  from utils.factors import compute_all_factors")
    print("  from utils.signal_prices import build_signal_price_df")
    print("")
    print("  # Signal prices 생성")
    print("  signal_df = build_signal_price_df(prices, cfg)")
    print("  signal_df = signal_df.reindex(prices.index).ffill()")
    print("")
    print("  # Factors 계산 (signal prices 사용)")
    print("  factors = compute_all_factors(prices, signal_df)")
    print("")
    print("  # FV4 엔진 실행")
    print("  engine = FactorValueV4(FV4Config())")
    print("  weights = engine.build_portfolio(prices, factors, rebalance_dates)")
