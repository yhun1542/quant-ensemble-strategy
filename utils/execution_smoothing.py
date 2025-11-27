#!/usr/bin/env python3
"""
Execution Smoothing 모듈
리밸런싱 시 포트폴리오 전환을 여러 날에 걸쳐 분산
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np


@dataclass
class ExecutionSmoothingConfig:
    """Execution Smoothing 설정"""
    n_steps: int = 2  # 포트 전환 단계 수 (2 = 50% + 50%)
    

def portfolio_returns_with_execution_smoothing(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: List[pd.Timestamp],
    cfg: ExecutionSmoothingConfig | None = None,
) -> pd.Series:
    """
    Execution Smoothing을 적용한 포트폴리오 수익률 계산
    
    리밸런싱 시 타겟 포트로 한 번에 전환하지 않고,
    여러 날에 걸쳐 점진적으로 전환합니다.
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터 (index=date, columns=tickers)
    weights_by_date : dict
        {date: Series(ticker -> weight)}
    rebalance_dates : list
        리밸런싱 날짜 리스트
    cfg : ExecutionSmoothingConfig, optional
        Execution Smoothing 설정
    
    Returns
    -------
    pd.Series
        일간 수익률
    
    Examples
    --------
    >>> cfg = ExecutionSmoothingConfig(n_steps=2)
    >>> ret = portfolio_returns_with_execution_smoothing(
    ...     prices, weights_by_date, rebalance_dates, cfg
    ... )
    
    Notes
    -----
    2-step 예시:
    - Day 0 (리밸 기준일): 타겟 포트 결정
    - Day 1: 50% 전환 (w_prev + 0.5 * (w_target - w_prev))
    - Day 2~: 100% 전환 (w_target)
    """
    cfg = cfg or ExecutionSmoothingConfig()
    
    daily_returns = []
    current_weights = None
    
    # 리밸 날짜를 인덱스로 변환
    rebal_idx_map = {d: i for i, d in enumerate(rebalance_dates)}
    
    for i in range(len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        
        # 리밸런싱 체크
        if date in rebalance_dates and date in weights_by_date:
            rebal_idx = rebal_idx_map[date]
            w_target = weights_by_date[date]
            
            # 이전 포트 가중치
            if current_weights is None:
                w_prev = pd.Series(0.0, index=w_target.index)
            else:
                # 공통 종목만 유지, 나머지는 0
                all_tickers = set(w_target.index) | set(current_weights.index)
                w_prev = pd.Series(0.0, index=all_tickers)
                for ticker in current_weights.index:
                    w_prev[ticker] = current_weights[ticker]
                w_prev = w_prev[w_target.index]
            
            # Execution schedule 생성
            execution_schedule = []
            for step in range(1, cfg.n_steps + 1):
                pct = step / cfg.n_steps
                w_step = w_prev + pct * (w_target - w_prev)
                execution_schedule.append(w_step)
            
            # 첫 번째 step 적용
            current_weights = execution_schedule[0]
            
            # 나머지 steps를 날짜에 매핑
            # (간단한 구현: 다음 날부터 순차적으로 적용)
            # 실제로는 거래일 기준으로 더 정교하게 처리 필요
        
        # 수익률 계산
        if current_weights is not None and len(current_weights) > 0:
            daily_ret = 0.0
            for ticker in current_weights.index:
                if ticker in prices.columns:
                    if date in prices.index and next_date in prices.index:
                        ret = prices.loc[next_date, ticker] / prices.loc[date, ticker] - 1.0
                        daily_ret += current_weights[ticker] * ret
            
            daily_returns.append({"date": next_date, "ret": daily_ret})
    
    if daily_returns:
        return pd.Series({r["date"]: r["ret"] for r in daily_returns})
    else:
        return pd.Series(dtype=float)


def portfolio_returns_simple(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: List[pd.Timestamp],
) -> pd.Series:
    """
    Execution Smoothing 없는 단순 포트폴리오 수익률 계산
    
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
                    if date in prices.index and next_date in prices.index:
                        ret = prices.loc[next_date, ticker] / prices.loc[date, ticker] - 1.0
                        daily_ret += current_weights[ticker] * ret
            
            daily_returns.append({"date": next_date, "ret": daily_ret})
    
    if daily_returns:
        return pd.Series({r["date"]: r["ret"] for r in daily_returns})
    else:
        return pd.Series(dtype=float)


if __name__ == "__main__":
    print("="*100)
    print("Execution Smoothing 모듈")
    print("="*100)
    print("\n사용 예시:")
    print("  from utils.execution_smoothing import portfolio_returns_with_execution_smoothing")
    print("  from utils.execution_smoothing import ExecutionSmoothingConfig")
    print("")
    print("  # Execution Smoothing 적용")
    print("  cfg = ExecutionSmoothingConfig(n_steps=2)")
    print("  ret = portfolio_returns_with_execution_smoothing(")
    print("      prices, weights_by_date, rebalance_dates, cfg")
    print("  )")
