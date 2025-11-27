"""
Factor Quality-Value Engine v2.1

Improvements over v2.0:
- Inverse-Vol Weighting option
- Configurable top quantile (0.2 ~ 0.4)
- Better portfolio diversification
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from utils.fundamental_factors import compute_value_score, compute_quality_score


class FactorQVEngineV21:
    """Quality-Value Engine with Inverse-Vol Weighting"""
    
    def __init__(
        self,
        top_quantile: float = 0.3,
        long_gross: float = 1.0,
        short_gross: float = 0.0,
        long_only: bool = True,
        use_inverse_vol: bool = True,
        vol_lookback: int = 63,
    ):
        """
        Parameters:
        -----------
        top_quantile : float
            Fraction of universe to select (0.2 = top 20%, 0.3 = top 30%)
        long_gross : float
            Target gross exposure for long leg
        short_gross : float
            Target gross exposure for short leg
        long_only : bool
            If True, only long positions
        use_inverse_vol : bool
            If True, use inverse-volatility weighting instead of equal weight
        vol_lookback : int
            Lookback period for volatility calculation (days)
        """
        self.top_quantile = top_quantile
        self.long_gross = long_gross
        self.short_gross = short_gross
        self.long_only = long_only
        self.use_inverse_vol = use_inverse_vol
        self.vol_lookback = vol_lookback
    
    def build_signals(self, fund_daily: pd.DataFrame) -> pd.Series:
        """
        Build QV score from fundamental data
        
        Parameters:
        -----------
        fund_daily : pd.DataFrame
            MultiIndex (date, ticker), columns = fundamental indicators
        
        Returns:
        --------
        qv_score : pd.Series
            MultiIndex (date, ticker), cross-sectional z-scored QV scores
        """
        value = compute_value_score(fund_daily)
        quality = compute_quality_score(fund_daily)
        
        # 50:50 combination
        qv_raw = 0.5 * quality + 0.5 * value
        
        # Cross-sectional z-score
        def _zscore(x: pd.Series) -> pd.Series:
            std = x.std(ddof=0)
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=x.index)
            return (x - x.mean()) / std
        
        qv = qv_raw.groupby(level=0).transform(_zscore)
        return qv.rename("qv_score")
    
    def build_portfolio(
        self,
        fund_daily: pd.DataFrame,
        prices: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Build portfolio weights with optional inverse-vol weighting
        
        Parameters:
        -----------
        fund_daily : pd.DataFrame
            MultiIndex (date, ticker), fundamental data
        prices : pd.DataFrame
            index=date, columns=tickers, adjusted close prices
        rebalance_dates : list[pd.Timestamp]
            Rebalancing dates
        
        Returns:
        --------
        weights_by_date : Dict[pd.Timestamp, pd.Series]
            {date: Series(index=ticker, values=weight)}
        """
        # Build signals
        qv = self.build_signals(fund_daily)
        
        # Calculate volatility
        ret = prices.pct_change()
        vol = ret.rolling(self.vol_lookback).std()
        
        weights_by_date: Dict[pd.Timestamp, pd.Series] = {}
        
        for d in rebalance_dates:
            # Check data availability
            if d not in qv.index.get_level_values(0):
                continue
            if d not in vol.index:
                continue
            
            # Get cross-section
            cs = qv.loc[d].dropna()
            if cs.empty:
                continue
            
            n = len(cs)
            n_long = max(int(n * self.top_quantile), 1)
            n_short = max(int(n * self.top_quantile), 1)
            
            # Select top/bottom
            cs_sorted = cs.sort_values(ascending=False)
            long_names = cs_sorted.head(n_long).index
            short_names = cs_sorted.tail(n_short).index
            
            portfolio: dict[str, float] = {}
            
            # ===== Long leg =====
            if self.use_inverse_vol:
                vols_long = vol.loc[d, long_names]
                # Handle NaN/inf
                vols_long = vols_long.replace([0, np.inf, -np.inf], np.nan).dropna()
                if vols_long.empty:
                    continue
                
                inv_long = 1.0 / vols_long
                w_long_raw = inv_long
            else:
                w_long_raw = pd.Series(1.0, index=long_names)
            
            # Normalize to long_gross
            w_long = w_long_raw / w_long_raw.sum() * self.long_gross
            portfolio.update(w_long.to_dict())
            
            # ===== Short leg (optional) =====
            if not self.long_only and self.short_gross > 0:
                if self.use_inverse_vol:
                    vols_short = vol.loc[d, short_names]
                    vols_short = vols_short.replace([0, np.inf, -np.inf], np.nan).dropna()
                    if not vols_short.empty:
                        inv_short = 1.0 / vols_short
                        w_short_raw = inv_short
                        # Normalize to short_gross (negative)
                        w_short = -w_short_raw / w_short_raw.sum() * self.short_gross
                        for tkr, w in w_short.items():
                            portfolio[tkr] = portfolio.get(tkr, 0.0) + w
                else:
                    w_short_raw = pd.Series(1.0, index=short_names)
                    w_short = -w_short_raw / w_short_raw.sum() * self.short_gross
                    for tkr, w in w_short.items():
                        portfolio[tkr] = portfolio.get(tkr, 0.0) + w
            
            if portfolio:
                weights_by_date[d] = pd.Series(portfolio)
        
        return weights_by_date
