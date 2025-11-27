"""
Fundamental Factor Calculations: Value and Quality Scores
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def xsec_zscore(s: pd.Series, winsor_pct: float = 0.01) -> pd.Series:
    """
    Cross-sectional z-score with winsorization for MultiIndex (date, ticker) Series.
    
    This function:
    1. Groups by date
    2. Removes inf values
    3. Winsorizes at specified percentiles
    4. Calculates z-score within each date
    5. Fills NaN with 0
    
    Args:
        s: MultiIndex (date, ticker) Series
        winsor_pct: Percentile for winsorization (0.01 = 1%)
    
    Returns:
        Z-scored Series with same index
    """
    # Remove inf values
    s = s.replace([np.inf, -np.inf], np.nan)
    
    # Group by date (level 0)
    def _winsor_zscore(x: pd.Series) -> pd.Series:
        # Winsorize
        lower = x.quantile(winsor_pct)
        upper = x.quantile(1 - winsor_pct)
        x_clipped = x.clip(lower, upper)
        
        # Calculate z-score
        mean = x_clipped.mean()
        std = x_clipped.std(ddof=0)
        
        if std == 0 or pd.isna(std):
            return x_clipped * 0.0  # Return zeros with same index
        
        return (x_clipped - mean) / std
    
    out = s.groupby(level=0, group_keys=False).apply(_winsor_zscore)
    out = out.fillna(0.0)
    
    return out


def compute_value_score(fund_daily: pd.DataFrame) -> pd.Series:
    """
    Compute Value Score from fundamental indicators.
    
    Value indicators (lower is better):
    - P/E Ratio (pe): Price / Earnings
    - P/B Ratio (pb): Price / Book Value
    - P/S Ratio (ps): Price / Sales
    - EV/EBITDA (evebitda): Enterprise Value / EBITDA
    
    Args:
        fund_daily: MultiIndex (date, ticker) DataFrame with columns:
                   ['pe', 'pb', 'ps', 'evebitda', ...]
    
    Returns:
        MultiIndex (date, ticker) Series with value_score
    """
    pe       = fund_daily["pe"]
    pb       = fund_daily["pb"]
    ps       = fund_daily["ps"]
    evebitda = fund_daily["evebitda"]
    
    # Invert direction: lower ratios = higher value
    z_pe = xsec_zscore(-pe)
    z_pb = xsec_zscore(-pb)
    z_ps = xsec_zscore(-ps)
    z_ev = xsec_zscore(-evebitda)
    
    # Equal-weighted combination
    value_raw = 0.25 * z_pe + 0.25 * z_pb + 0.25 * z_ps + 0.25 * z_ev
    
    # Final z-score normalization
    value = xsec_zscore(value_raw)
    
    return value.rename("value_score")


def compute_quality_score(fund_daily: pd.DataFrame) -> pd.Series:
    """
    Compute Quality Score from fundamental indicators.
    
    Quality indicators:
    - ROE (roe): Return on Equity (higher is better)
    - EBITDA Margin (ebitdamargin): Operating efficiency (higher is better)
    - Debt-to-Equity (de): Leverage (lower is better)
    - Current Ratio (currentratio): Liquidity (higher is better)
    
    Args:
        fund_daily: MultiIndex (date, ticker) DataFrame with columns:
                   ['roe', 'ebitdamargin', 'de', 'currentratio', ...]
    
    Returns:
        MultiIndex (date, ticker) Series with quality_score
    """
    roe        = fund_daily["roe"]
    op_mgn     = fund_daily["ebitdamargin"]
    d2e        = fund_daily["de"]
    curr_ratio = fund_daily["currentratio"]
    
    # Z-score with correct direction
    z_roe  = xsec_zscore(roe)          # Higher is better
    z_mgn  = xsec_zscore(op_mgn)       # Higher is better
    z_lev  = xsec_zscore(-d2e)         # Lower is better
    z_liq  = xsec_zscore(curr_ratio)   # Higher is better
    
    # Weighted combination (adjusted without intcov)
    quality_raw = (
        0.35 * z_roe +    # Profitability (35%)
        0.25 * z_mgn +    # Operating efficiency (25%)
        0.25 * z_lev +    # Financial health (25%)
        0.15 * z_liq      # Short-term liquidity (15%)
    )
    
    # Final z-score normalization
    quality = xsec_zscore(quality_raw)
    
    return quality.rename("quality_score")
