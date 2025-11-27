#!/usr/bin/env python3
"""
Cross-sectional Momentum Engine v2 (Lookahead-Free)
- 룩어헤드 바이어스 제거
- Universe: 30 mega-cap stocks
- Rebalancing: Monthly
- Long-only, top 6 names
"""
import json
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd


# ===== 유틸 함수들 =====

def load_price_data(
    path: str = "data/price_data_sp500.csv",
    universe: List[str] = None,
) -> pd.DataFrame:
    """
    가격 데이터 로딩
    - index: DatetimeIndex (거래일)
    - columns: ticker
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    if universe is not None:
        cols = [c for c in universe if c in prices.columns]
        prices = prices[cols]

    return prices


def get_monthly_rebalance_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    월간 리밸 날짜: 각 달의 첫 번째 거래일
    """
    df = pd.DataFrame(index=index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    first_days = df.groupby(["year", "month"], group_keys=False).apply(lambda x: x.index.min())
    return list(first_days.sort_values())


def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: List[pd.Timestamp],
) -> pd.Series:
    """
    리밸 날짜별 가중치를 받아 일간 포트 수익률 계산
    - 가정: 리밸일 종가 기준으로 다음날부터 적용
    """
    prices = prices.sort_index()

    # 리밸 날짜를 실제 거래일 중 있는 날짜로만 필터
    rebalance_dates = [d for d in rebalance_dates if d in prices.index]

    # 다음 리밸일까지 구간 슬라이싱하면서 수익률 계산
    ret_list = []

    for i, d in enumerate(rebalance_dates):
        w = weights_by_date.get(d)
        if w is None or len(w) == 0:
            continue

        # 현재 리밸 날짜부터 다음 리밸 전날까지
        if i < len(rebalance_dates) - 1:
            d_next = rebalance_dates[i + 1]
            idx = (prices.index > d) & (prices.index <= d_next)
        else:
            idx = prices.index > d

        # 해당 구간 가격
        px = prices.loc[idx, w.index]
        if px.shape[0] == 0:
            continue

        # 일간 종목 수익률
        daily_ret = px.pct_change().fillna(0.0)

        # 포트 일간 수익률
        port_ret = (daily_ret * w.values).sum(axis=1)
        ret_list.append(port_ret)

    if not ret_list:
        return pd.Series(dtype=float)

    ret = pd.concat(ret_list).sort_index()
    ret.name = "ret_mom_cs_v2"
    return ret


# ===== 모멘텀 엔진 본체 =====

@dataclass
class MomentumCSEngineV2Config:
    lookback_long: int = 252        # 장기 모멘텀 기간
    lookback_exclude: int = 21      # 최근 제외 기간
    lookback_short: int = 21        # 단기 모멘텀 기간
    n_long: int = 6                 # 최종 롱 종목 수
    long_top_pct: float = 0.3       # 장기 모멘텀 상위 퍼센트
    short_overheat_pct: float = 0.1 # 단기 과열 구간 퍼센트


class MomentumCSEngineV2:
    """
    Cross-sectional momentum engine (v2) - Lookahead-Free
    - Universe: 30 mega-cap stocks
    - Rebal: monthly (first trading day)
    - Long-only, top 6 names
    - 룩어헤드 바이어스 제거
    """

    def __init__(self, config: MomentumCSEngineV2Config = None):
        self.config = config or MomentumCSEngineV2Config()

    def compute_momentum_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        prices: DataFrame, index=date, columns=tickers
        return: MultiIndex (date, ticker) DataFrame with:
            - mom_252_ex_21: P(t-21) / P(t-252) - 1
            - mom_21: P(t) / P(t-21) - 1
        
        **룩어헤드 방지**:
        - t 시점에서 계산할 때, t 이전의 데이터만 사용
        - shift()를 사용하여 과거 데이터 참조
        """
        cfg = self.config

        # 장기 모멘텀 (최근 lookback_exclude 제외)
        # mom_252_ex_21 = P(t-21) / P(t-252) - 1
        # 
        # shift(21)은 t 시점에서 t-21 가격을 가져옴
        # shift(252)는 t 시점에서 t-252 가격을 가져옴
        price_t_minus_21 = prices.shift(cfg.lookback_exclude)
        price_t_minus_252 = prices.shift(cfg.lookback_long)
        mom_252_ex_21 = price_t_minus_21 / price_t_minus_252 - 1.0

        # 단기 모멘텀
        # mom_21 = P(t) / P(t-21) - 1
        price_t_minus_21_short = prices.shift(cfg.lookback_short)
        mom_21 = prices / price_t_minus_21_short - 1.0

        df_long = mom_252_ex_21.stack().to_frame("mom_252_ex_21")
        df_short = mom_21.stack().to_frame("mom_21")

        factors = df_long.join(df_short, how="inner")
        return factors  # index: (date, ticker)

    def select_portfolio_on_date(self, factors_cs: pd.DataFrame) -> pd.Series:
        """
        factors_cs: 해당 날짜의 cross-section
            index=ticker, columns=[mom_252_ex_21, mom_21]
        return: weight Series (index=ticker, values=weights, sum=1.0)
        """
        cfg = self.config
        n_total = len(factors_cs)
        if n_total == 0:
            return pd.Series(dtype=float)

        df = factors_cs.copy().dropna()

        if df.empty:
            return pd.Series(dtype=float)

        # 1) 장기 모멘텀 상위 long_top_pct 필터
        k_long = max(int(n_total * cfg.long_top_pct), cfg.n_long)
        df_sorted_long = df.sort_values("mom_252_ex_21", ascending=False)
        df_long = df_sorted_long.iloc[:k_long]

        # 2) 단기 과열 상위 short_overheat_pct 제거
        k_over = max(int(n_total * cfg.short_overheat_pct), 1)
        overheat_names = (
            df.sort_values("mom_21", ascending=False).iloc[:k_over].index
        )
        df_final = df_long[~df_long.index.isin(overheat_names)]

        if df_final.empty:
            df_final = df_long  # 전부 과열에 걸렸으면 일단 장기 상위 그대로 사용

        # 3) 최종 상위 n_long 선택
        df_final = df_final.sort_values("mom_252_ex_21", ascending=False)
        selected = df_final.iloc[: cfg.n_long]

        if selected.empty:
            return pd.Series(dtype=float)

        # 4) equal weight
        w = pd.Series(1.0 / len(selected), index=selected.index, name="weight")
        return w

    def build_portfolio(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        월간 리밸 기준 포트 구성 히스토리 생성
        return:
            - weights_by_date: {rebalance_date: Series(weights)}
        """
        factors = self.compute_momentum_factors(prices)
        weights_by_date: Dict[pd.Timestamp, pd.Series] = {}

        # factors index: (date, ticker)
        for d in rebalance_dates:
            try:
                cs = factors.loc[d].dropna()
            except KeyError:
                continue

            w = self.select_portfolio_on_date(cs)
            if len(w) == 0:
                continue

            weights_by_date[d] = w

        return weights_by_date


# ===== 스크립트 실행부 =====

def calc_monthly_metrics(ret_daily: pd.Series) -> dict:
    """
    일간 수익률 기준으로 월간 성과 계산
    """
    ret_daily = ret_daily.sort_index()
    monthly_ret = ret_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    years = (monthly_ret.index[-1] - monthly_ret.index[0]).days / 365.25
    if years <= 0:
        return {}

    cum_ret = (1 + monthly_ret).prod()
    ann_ret = cum_ret ** (1 / years) - 1

    vol_m = monthly_ret.std(ddof=0)
    ann_vol = vol_m * np.sqrt(12)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # MaxDD (월간 기준)
    wealth = (1 + monthly_ret).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_dd = dd.min()

    win_rate = (monthly_ret > 0).mean()

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
    }


def main():
    print("=" * 100)
    print("Momentum CS v2 Engine (Lookahead-Free) - 30 Mega-cap Universe")
    print("=" * 100)
    
    # 1) 유니버스 정의 (현재 30개 메가캡)
    universe_30 = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
    ]

    # 2) 가격 데이터 로딩
    prices = load_price_data("data/price_data_sp500.csv", universe_30)
    
    print(f"\n가격 데이터:")
    print(f"  - 기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"  - 거래일 수: {len(prices)}")
    print(f"  - 종목 수: {len(prices.columns)}")

    # 3) 리밸 날짜 (월간)
    rebalance_dates = get_monthly_rebalance_dates(prices.index)
    print(f"  - 리밸런싱 횟수: {len(rebalance_dates)}")

    # 4) 모멘텀 엔진으로 포트 구성
    engine = MomentumCSEngineV2()
    weights_by_date = engine.build_portfolio(prices, rebalance_dates)
    print(f"  - 포트폴리오 구성 완료: {len(weights_by_date)}개 리밸런싱 날짜")

    # 5) 일간 수익률 계산
    ret_daily = portfolio_returns_from_weights(prices, weights_by_date, rebalance_dates)
    print(f"  - 일간 수익률 계산 완료: {len(ret_daily)}일")

    # 6) 월간 성과 메트릭 계산
    metrics = calc_monthly_metrics(ret_daily)

    # 7) 결과 저장 (JSON)
    out = {
        "metrics": metrics,
        "daily_returns": {
            "index": [d.strftime("%Y-%m-%d") for d in ret_daily.index],
            "values": ret_daily.astype(float).tolist(),
        },
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "momentum_cs_v2_fixed_oos.json"
    
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✅ 결과 저장 완료: {output_path}")
    
    print("\n" + "=" * 100)
    print("성과 지표 (월간 기준)")
    print("=" * 100)
    print(f"Sharpe Ratio:        {metrics['sharpe']:.4f}")
    print(f"Annual Return:       {metrics['ann_return']*100:.2f}%")
    print(f"Annual Volatility:   {metrics['ann_vol']*100:.2f}%")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
