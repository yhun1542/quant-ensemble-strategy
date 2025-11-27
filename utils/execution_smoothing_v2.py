#!/usr/bin/env python3
"""
Execution Smoothing v2 모듈
리밸런싱 시 포트폴리오 전환을 여러 거래일에 걸쳐 분산
AI 평가 피드백 반영: 거래일 캘린더 처리 + 로깅 + 에러 처리
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSmoothingConfig:
    """Execution Smoothing 설정"""
    n_steps: int = 2  # 포트 전환 단계 수 (2 = 50% + 50%)
    

def find_next_trading_days(
    start_date: pd.Timestamp,
    all_trading_days: pd.DatetimeIndex,
    n_days: int
) -> List[pd.Timestamp]:
    """
    시작일 이후 N개의 거래일 찾기
    
    Parameters
    ----------
    start_date : pd.Timestamp
        시작 날짜
    all_trading_days : pd.DatetimeIndex
        모든 거래일 인덱스
    n_days : int
        필요한 거래일 수
    
    Returns
    -------
    list of pd.Timestamp
        N개의 거래일 (부족하면 가능한 만큼만 반환)
    """
    try:
        start_idx = all_trading_days.get_loc(start_date)
    except KeyError:
        # start_date가 거래일이 아닌 경우, 다음 거래일 찾기
        future_dates = all_trading_days[all_trading_days > start_date]
        if len(future_dates) == 0:
            logger.warning(f"No trading days after {start_date}")
            return []
        start_idx = all_trading_days.get_loc(future_dates[0])
    
    # 다음 N개 거래일
    next_days = all_trading_days[start_idx + 1: start_idx + 1 + n_days]
    
    if len(next_days) < n_days:
        logger.warning(
            f"Only {len(next_days)} trading days available after {start_date} "
            f"(requested {n_days})"
        )
    
    return list(next_days)


def portfolio_returns_with_execution_smoothing(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: List[pd.Timestamp],
    cfg: ExecutionSmoothingConfig | None = None,
) -> pd.Series:
    """
    Execution Smoothing을 적용한 포트폴리오 수익률 계산 (v2)
    
    개선 사항:
    - 거래일 캘린더 처리 (주말/휴장 자동 스킵)
    - 로깅 추가
    - 에러 처리 강화
    
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
    2-step 예시 (거래일 기준):
    - Day 0 (리밸 기준일): 타겟 포트 결정
    - Day 1 (다음 거래일): 50% 전환
    - Day 2 (그 다음 거래일): 100% 전환
    """
    cfg = cfg or ExecutionSmoothingConfig()
    
    if len(prices) == 0:
        logger.error("Empty prices DataFrame")
        return pd.Series(dtype=float)
    
    if len(rebalance_dates) == 0:
        logger.warning("No rebalance dates provided")
        return pd.Series(dtype=float)
    
    # 거래일 인덱스
    trading_days = prices.index
    
    # 리밸 날짜별 실행 스케줄 생성
    execution_calendar = {}  # {date: weights}
    
    for rebal_date in rebalance_dates:
        if rebal_date not in weights_by_date:
            logger.warning(f"No weights for rebalance date {rebal_date}")
            continue
        
        w_target = weights_by_date[rebal_date]
        
        # 이전 포트 가중치 (첫 리밸이면 0)
        prev_dates = [d for d in execution_calendar.keys() if d < rebal_date]
        if prev_dates:
            w_prev = execution_calendar[max(prev_dates)]
            # 공통 종목 정렬
            all_tickers = set(w_target.index) | set(w_prev.index)
            w_prev_aligned = pd.Series(0.0, index=w_target.index)
            for ticker in w_prev.index:
                if ticker in w_target.index:
                    w_prev_aligned[ticker] = w_prev[ticker]
            w_prev = w_prev_aligned
        else:
            w_prev = pd.Series(0.0, index=w_target.index)
        
        # 실행 스케줄 생성
        execution_schedule = []
        for step in range(1, cfg.n_steps + 1):
            pct = step / cfg.n_steps
            w_step = w_prev + pct * (w_target - w_prev)
            execution_schedule.append(w_step)
        
        # 다음 N개 거래일 찾기
        exec_dates = find_next_trading_days(rebal_date, trading_days, cfg.n_steps)
        
        # 실행 캘린더에 추가
        for i, exec_date in enumerate(exec_dates):
            if i < len(execution_schedule):
                execution_calendar[exec_date] = execution_schedule[i]
                logger.debug(
                    f"Rebal {rebal_date} -> Exec {exec_date} (step {i+1}/{cfg.n_steps})"
                )
        
        # 마지막 step이 모두 실행되지 않은 경우, 마지막 가용 날짜에 타겟 포트 적용
        if len(exec_dates) < cfg.n_steps and len(exec_dates) > 0:
            execution_calendar[exec_dates[-1]] = w_target
            logger.info(
                f"Rebal {rebal_date}: incomplete execution, "
                f"applying target weights on {exec_dates[-1]}"
            )
    
    # 일간 수익률 계산
    daily_returns = []
    current_weights = None
    
    for i in range(len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        
        # 실행 캘린더 체크
        if date in execution_calendar:
            current_weights = execution_calendar[date]
        
        # 수익률 계산
        if current_weights is not None and len(current_weights) > 0:
            try:
                daily_ret = 0.0
                for ticker in current_weights.index:
                    if ticker in prices.columns:
                        if date in prices.index and next_date in prices.index:
                            p_curr = prices.loc[date, ticker]
                            p_next = prices.loc[next_date, ticker]
                            
                            # NaN 체크
                            if pd.isna(p_curr) or pd.isna(p_next) or p_curr == 0:
                                logger.warning(
                                    f"Invalid price for {ticker} on {date}/{next_date}"
                                )
                                continue
                            
                            ret = p_next / p_curr - 1.0
                            daily_ret += current_weights[ticker] * ret
                
                daily_returns.append({"date": next_date, "ret": daily_ret})
            
            except Exception as e:
                logger.error(f"Error calculating return on {date}: {e}")
                continue
    
    if daily_returns:
        result = pd.Series({r["date"]: r["ret"] for r in daily_returns})
        logger.info(
            f"Execution smoothing complete: {len(result)} days, "
            f"avg return {result.mean():.4f}"
        )
        return result
    else:
        logger.warning("No returns calculated")
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
            try:
                daily_ret = 0.0
                for ticker in current_weights.index:
                    if ticker in prices.columns:
                        if date in prices.index and next_date in prices.index:
                            p_curr = prices.loc[date, ticker]
                            p_next = prices.loc[next_date, ticker]
                            
                            if pd.isna(p_curr) or pd.isna(p_next) or p_curr == 0:
                                continue
                            
                            ret = p_next / p_curr - 1.0
                            daily_ret += current_weights[ticker] * ret
                
                daily_returns.append({"date": next_date, "ret": daily_ret})
            
            except Exception as e:
                logger.error(f"Error calculating return on {date}: {e}")
                continue
    
    if daily_returns:
        return pd.Series({r["date"]: r["ret"] for r in daily_returns})
    else:
        return pd.Series(dtype=float)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*100)
    print("Execution Smoothing v2 모듈")
    print("="*100)
    print("\n개선 사항:")
    print("  - 거래일 캘린더 처리 (주말/휴장 자동 스킵)")
    print("  - 로깅 추가 (디버깅 용이)")
    print("  - 에러 처리 강화 (NaN, 0 가격 처리)")
    print("\n사용 예시:")
    print("  from utils.execution_smoothing_v2 import portfolio_returns_with_execution_smoothing")
    print("  from utils.execution_smoothing_v2 import ExecutionSmoothingConfig")
    print("")
    print("  # Execution Smoothing 적용")
    print("  cfg = ExecutionSmoothingConfig(n_steps=2)")
    print("  ret = portfolio_returns_with_execution_smoothing(")
    print("      prices, weights_by_date, rebalance_dates, cfg")
    print("  )")
