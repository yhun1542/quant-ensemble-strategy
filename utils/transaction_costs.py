#!/usr/bin/env python3
"""
Transaction Costs Module

This module calculates realistic transaction costs for portfolio rebalancing.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Transaction cost model for US equities.
    
    Includes:
    - Commission costs
    - Bid-ask spread
    - Market impact
    """
    
    def __init__(
        self,
        commission_bps: float = 0.5,
        spread_bps: float = 5.0,
        impact_bps: float = 3.0
    ):
        """
        Initialize transaction cost model.
        
        Parameters
        ----------
        commission_bps : float
            Commission cost in basis points (default: 0.5 bps = $0.005 per share)
        spread_bps : float
            Bid-ask spread cost in basis points (default: 5 bps for liquid stocks)
        impact_bps : float
            Market impact cost in basis points (default: 3 bps for small orders)
        """
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.impact_bps = impact_bps
        self.total_bps = commission_bps + spread_bps + impact_bps
        
        logger.info(
            f"Transaction cost model initialized: "
            f"{self.total_bps:.1f} bps total "
            f"(commission {commission_bps:.1f}, spread {spread_bps:.1f}, impact {impact_bps:.1f})"
        )
    
    def calculate_turnover(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Parameters
        ----------
        old_weights : pd.Series
            Previous portfolio weights
        new_weights : pd.Series
            New portfolio weights
        
        Returns
        -------
        float
            Turnover (sum of absolute weight changes)
        """
        # Align weights
        all_tickers = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_tickers).fillna(0.0)
        new_aligned = new_weights.reindex(all_tickers).fillna(0.0)
        
        # Turnover = sum of absolute changes
        turnover = (new_aligned - old_aligned).abs().sum()
        
        return turnover
    
    def calculate_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series
    ) -> float:
        """
        Calculate transaction cost for rebalancing.
        
        Parameters
        ----------
        old_weights : pd.Series
            Previous portfolio weights
        new_weights : pd.Series
            New portfolio weights
        
        Returns
        -------
        float
            Transaction cost as a fraction of portfolio value
        """
        turnover = self.calculate_turnover(old_weights, new_weights)
        
        # Cost = turnover × cost per trade (in bps)
        cost = turnover * (self.total_bps / 10000)
        
        logger.debug(
            f"Turnover: {turnover:.2%}, Cost: {cost:.4%} "
            f"({self.total_bps:.1f} bps)"
        )
        
        return cost
    
    def apply_costs_to_returns(
        self,
        returns: pd.Series,
        weights_by_date: Dict[pd.Timestamp, pd.Series],
        rebalance_dates: list[pd.Timestamp]
    ) -> pd.Series:
        """
        Apply transaction costs to portfolio returns.
        
        Parameters
        ----------
        returns : pd.Series
            Gross daily returns
        weights_by_date : dict
            Portfolio weights by date
        rebalance_dates : list
            List of rebalance dates
        
        Returns
        -------
        pd.Series
            Net daily returns (after transaction costs)
        """
        net_returns = returns.copy()
        
        # Track previous weights
        prev_weights = None
        total_costs = 0.0
        
        for i, rebal_date in enumerate(rebalance_dates):
            if rebal_date not in weights_by_date:
                continue
            
            new_weights = weights_by_date[rebal_date]
            
            # Calculate cost (skip first rebalance)
            if prev_weights is not None:
                cost = self.calculate_cost(prev_weights, new_weights)
                total_costs += cost
                
                # Find the next trading day after rebalance
                future_dates = returns.index[returns.index > rebal_date]
                if len(future_dates) > 0:
                    next_day = future_dates[0]
                    
                    # Deduct cost from next day's return
                    if next_day in net_returns.index:
                        net_returns.loc[next_day] -= cost
                        
                        logger.debug(
                            f"Rebalance {rebal_date.date()}: "
                            f"Cost {cost:.4%} deducted on {next_day.date()}"
                        )
            
            prev_weights = new_weights
        
        logger.info(
            f"Total transaction costs: {total_costs:.2%} "
            f"({len(rebalance_dates)} rebalances)"
        )
        
        return net_returns


def apply_transaction_costs(
    returns: pd.Series,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: list[pd.Timestamp],
    cost_bps: float = 8.5
) -> pd.Series:
    """
    Convenience function to apply transaction costs.
    
    Parameters
    ----------
    returns : pd.Series
        Gross daily returns
    weights_by_date : dict
        Portfolio weights by date
    rebalance_dates : list
        List of rebalance dates
    cost_bps : float
        Total transaction cost in basis points (default: 8.5 bps)
    
    Returns
    -------
    pd.Series
        Net daily returns (after transaction costs)
    """
    # Split total cost into components (approximate)
    commission_bps = cost_bps * 0.06  # ~6% of total
    spread_bps = cost_bps * 0.59      # ~59% of total
    impact_bps = cost_bps * 0.35      # ~35% of total
    
    model = TransactionCostModel(
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        impact_bps=impact_bps
    )
    
    return model.apply_costs_to_returns(returns, weights_by_date, rebalance_dates)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*100)
    print("Transaction Costs Module Test")
    print("="*100)
    
    # Create sample weights
    old_w = pd.Series({'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4})
    new_w = pd.Series({'AAPL': 0.2, 'GOOGL': 0.5, 'TSLA': 0.3})
    
    model = TransactionCostModel()
    
    turnover = model.calculate_turnover(old_w, new_w)
    cost = model.calculate_cost(old_w, new_w)
    
    print(f"\nTurnover: {turnover:.2%}")
    print(f"Cost: {cost:.4%} ({model.total_bps:.1f} bps)")
    
    print("\n" + "="*100)
    print("Test Complete")
    print("="*100)
