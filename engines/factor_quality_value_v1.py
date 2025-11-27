"""
Quality + Value Factor Engine v1

This engine combines fundamental Quality and Value factors to generate
long/short or long-only portfolios.
"""

from __future__ import annotations
from typing import Dict

import pandas as pd
import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

from utils.fundamental_factors import compute_value_score, compute_quality_score


class FactorQVEngineV1:
    """
    Quality + Value Engine v1
    
    Features:
    - Combines Quality and Value scores with 50:50 weighting
    - Supports both long-only and long-short strategies
    - Configurable gross exposure for long and short legs
    - Top quantile selection for long (high QV) and short (low QV)
    
    Args:
        top_quantile: Fraction of universe for long/short legs (e.g., 0.2 = top/bottom 20%)
        long_gross: Target gross exposure for long leg (e.g., 0.5 = 50%)
        short_gross: Target gross exposure for short leg (e.g., 0.5 = 50%)
        long_only: If True, only long positions (ignores short_gross)
    """
    
    def __init__(
        self,
        top_quantile: float = 0.2,
        long_gross: float = 0.5,
        short_gross: float = 0.5,
        long_only: bool = False,
    ):
        self.top_quantile = top_quantile
        self.long_gross   = long_gross
        self.short_gross  = short_gross
        self.long_only    = long_only
    
    def build_signals(self, fund_daily: pd.DataFrame) -> pd.Series:
        """
        Build QV signals from fundamental data.
        
        Args:
            fund_daily: MultiIndex (date, ticker) DataFrame with fundamental indicators
        
        Returns:
            MultiIndex (date, ticker) Series with qv_score
        """
        value   = compute_value_score(fund_daily)
        quality = compute_quality_score(fund_daily)
        
        # 50:50 combination
        qv_raw = 0.5 * quality + 0.5 * value
        
        # Cross-sectional z-score normalization
        def _z(x: pd.Series) -> pd.Series:
            std = x.std(ddof=0)
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=x.index)
            return (x - x.mean()) / std
        
        qv = qv_raw.groupby(level=0).transform(_z)
        
        return qv.rename("qv_score")
    
    def build_portfolio(
        self,
        fund_daily: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Build portfolio weights for each rebalance date.
        
        Args:
            fund_daily: MultiIndex (date, ticker) DataFrame with fundamental indicators
            rebalance_dates: List of rebalance dates
        
        Returns:
            Dictionary mapping rebalance dates to weight Series (ticker -> weight)
        """
        qv = self.build_signals(fund_daily)
        weights_by_date: Dict[pd.Timestamp, pd.Series] = {}
        
        for d in rebalance_dates:
            if d not in qv.index.get_level_values(0):
                continue
            
            cs = qv.loc[d].dropna()  # index=ticker, value=qv_score
            if cs.empty:
                continue
            
            n = len(cs)
            n_long  = max(int(n * self.top_quantile), 1)
            n_short = max(int(n * self.top_quantile), 1)
            
            cs_sorted   = cs.sort_values(ascending=False)
            long_names  = cs_sorted.head(n_long).index
            short_names = cs_sorted.tail(n_short).index
            
            # Long leg: equal-weighted, scaled to long_gross
            w_long_raw = pd.Series(1.0, index=long_names)
            w_long = w_long_raw / w_long_raw.sum() * self.long_gross
            portfolio: dict[str, float] = w_long.to_dict()
            
            # Short leg (optional)
            if not self.long_only and self.short_gross > 0:
                w_short_raw = pd.Series(1.0, index=short_names)
                w_short = -w_short_raw / w_short_raw.sum() * self.short_gross
                for tkr, w in w_short.items():
                    portfolio[tkr] = portfolio.get(tkr, 0.0) + w
            
            if portfolio:
                w = pd.Series(portfolio)
                
                # Optional: verify gross/net exposure
                gross = float(w.abs().sum())
                net   = float(w.sum())
                
                # Sanity check
                if self.long_only:
                    assert net > 0.99 * self.long_gross, f"Long-only net exposure too low: {net}"
                
                weights_by_date[d] = w
        
        return weights_by_date
