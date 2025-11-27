#!/usr/bin/env python3
"""
Portfolio Returns Calculation with Execution Smoothing

This module provides two versions of portfolio returns calculation:
1. Instant rebalance (v1.x baseline)
2. Execution Smoothing v2 (split-step rebalancing)
"""
import numpy as np
import pandas as pd
from typing import Dict


def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: list[pd.Timestamp],
) -> pd.Series:
    """
    v1.x baseline: Instant rebalance
    
    On rebalance date, portfolio jumps immediately to target weights.
    No execution smoothing.
    
    Args:
        prices: DataFrame of daily prices (index=date, columns=tickers)
        weights_by_date: Dict mapping rebalance dates to target weight Series
        rebalance_dates: List of rebalance dates
        
    Returns:
        Series of daily portfolio returns
    """
    daily_returns = []
    current_weights = None
    
    for i in range(len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        
        # Check for rebalance
        if date in rebalance_dates and date in weights_by_date:
            current_weights = weights_by_date[date]
        
        # Calculate return
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


def _blend_weights(
    prev_w: pd.Series | None,
    target_w: pd.Series,
    alpha: float,
    drop_tol: float = 1e-8,
) -> pd.Series:
    """
    Blend previous weights with target weights.
    
    Args:
        prev_w: Previous weight Series (None if no previous portfolio)
        target_w: Target weight Series
        alpha: Blending factor (0=prev, 1=target)
        drop_tol: Threshold for dropping near-zero weights
        
    Returns:
        Blended weight Series
    """
    if prev_w is None:
        # No previous portfolio → scale target by alpha
        blended = target_w * alpha
    else:
        # Blend: (1-alpha)*prev + alpha*target
        all_tickers = prev_w.index.union(target_w.index)
        prev = prev_w.reindex(all_tickers).fillna(0.0)
        tgt = target_w.reindex(all_tickers).fillna(0.0)
        blended = (1.0 - alpha) * prev + alpha * tgt
    
    # Drop near-zero weights
    blended = blended[blended.abs() > drop_tol]
    
    return blended


def portfolio_returns_with_execution_smoothing(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: list[pd.Timestamp],
    split_steps: int = 2,
) -> pd.Series:
    """
    Execution Smoothing v2: Split-step rebalancing
    
    Instead of jumping instantly to target weights, transition is split
    across multiple days to reduce rebalance date sensitivity and model
    realistic execution.
    
    Example (split_steps=2):
        Day 1: 50% prev + 50% target
        Day 2+: 100% target
    
    Args:
        prices: DataFrame of daily prices (index=date, columns=tickers)
        weights_by_date: Dict mapping rebalance dates to target weight Series
        rebalance_dates: List of rebalance dates
        split_steps: Number of steps to split rebalance (default=2)
        
    Returns:
        Series of daily portfolio returns
        
    Lookahead Prevention:
        - Target weights determined using data up to rebalance date close
        - Returns for date→next_date use blended weights
        - No future data leakage
    """
    prices = prices.sort_index()
    dates = prices.index
    
    daily_returns = []
    current_weights: pd.Series | None = None
    prev_weights: pd.Series | None = None
    target_weights: pd.Series | None = None
    step_idx: int | None = None
    
    # Convert rebalance_dates to set for faster lookup
    rebalance_set = set(rebalance_dates)
    
    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]
        
        # 1) Check for rebalance: update target_weights using data up to date close
        if date in rebalance_set and date in weights_by_date:
            prev_weights = current_weights
            target_weights = weights_by_date[date]
            step_idx = 0  # Start smoothing
        
        # 2) Determine weights for this period
        if target_weights is not None and step_idx is not None and step_idx < split_steps:
            # Smoothing in progress
            alpha = float(step_idx + 1) / float(split_steps)  # 1/split_steps, ..., 1
            blended_w = _blend_weights(prev_weights, target_weights, alpha)
            current_weights = blended_w
            step_idx += 1
        else:
            # Smoothing complete or no target → maintain current weights
            if target_weights is not None and step_idx is not None and step_idx >= split_steps:
                current_weights = target_weights
        
        # 3) Calculate return
        if current_weights is not None and len(current_weights) > 0:
            daily_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker in prices.columns:
                    px_now = prices.loc[date, ticker]
                    px_next = prices.loc[next_date, ticker]
                    if np.isfinite(px_now) and px_now > 0 and np.isfinite(px_next):
                        ret = px_next / px_now - 1.0
                        daily_ret += w * ret
            
            daily_returns.append({"date": next_date, "ret": daily_ret})
    
    if daily_returns:
        return pd.Series({r["date"]: r["ret"] for r in daily_returns}).sort_index()
    else:
        return pd.Series(dtype=float)
