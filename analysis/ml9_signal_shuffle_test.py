"""
ML9 Signal Shuffle Test

목표: ML9의 Sharpe 4.17이 진짜 알파인지, 구조적 바이어스인지 확인
방법: ML9 weights(시그널 대용)를 날짜별로 랜덤하게 섞어서 100회 반복
기대: 진짜 알파라면 Shuffle 후 Sharpe → 0 근처
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


# ====== 설정 ======
BASE_DIR = Path("/home/ubuntu/quant-ensemble-strategy")
N_TRIALS = 100


# ====== 데이터 로딩 ======

def load_prices() -> pd.DataFrame:
    """가격 데이터 로딩"""
    prices = pd.read_csv(BASE_DIR / "data" / "price_data_sp500.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date")
    prices.index = prices.index.tz_localize(None)
    
    TICKERS = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
    ]
    available_tickers = [t for t in TICKERS if t in prices.columns]
    prices = prices[available_tickers]
    
    return prices


def load_ml9_weights() -> Dict[pd.Timestamp, pd.Series]:
    """ML9 weights 로딩 (시그널 대용)"""
    with open(BASE_DIR / "results" / "ensemble_fv3c_ml9.json", "r") as f:
        data = json.load(f)
    
    ml9_weights = {}
    for date_str, weights_dict in data["ml9_weights"].items():
        if weights_dict:
            date = pd.to_datetime(date_str).tz_localize(None)
            ml9_weights[date] = pd.Series(weights_dict)
    
    return ml9_weights


# ====== 포트폴리오 수익률 계산 ======

def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
) -> pd.Series:
    """
    Weights 기반 포트폴리오 수익률 계산
    """
    prices = prices.sort_index()
    dates = prices.index
    
    weights_by_date_only = {}
    for dt, w in weights_by_date.items():
        date_only = dt.date()
        weights_by_date_only[date_only] = w
    
    current_weights = None
    daily_returns = []
    
    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]
        date_only = date.date()
        
        # 리밸일이면 weight 갱신
        if date_only in weights_by_date_only:
            current_weights = weights_by_date_only[date_only]
        
        if current_weights is None or current_weights.empty:
            continue
        
        # 일간 수익률 계산
        ret_cross = prices.loc[next_date] / prices.loc[date] - 1.0
        common_tickers = current_weights.index.intersection(ret_cross.index)
        
        if len(common_tickers) > 0:
            w = current_weights[common_tickers]
            r = ret_cross[common_tickers]
            port_ret = (w * r).sum()
            daily_returns.append({"date": next_date, "return": port_ret})
    
    if not daily_returns:
        return pd.Series(dtype=float)
    
    df_ret = pd.DataFrame(daily_returns).set_index("date")
    return df_ret["return"]


def calculate_sharpe(returns: pd.Series) -> float:
    """Sharpe Ratio 계산"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret == 0:
        return 0.0
    
    sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252))
    return sharpe


# ====== Signal Shuffle ======

def shuffle_weights(
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    seed: int
) -> Dict[pd.Timestamp, pd.Series]:
    """
    각 리밸런싱 날짜별로 weights를 ticker 간에 랜덤하게 섞기
    """
    rng = np.random.default_rng(seed)
    shuffled = {}
    
    for date, weights in weights_by_date.items():
        tickers = weights.index.tolist()
        values = weights.values.copy()
        
        # Shuffle values
        rng.shuffle(values)
        
        # 새로운 weights 생성
        shuffled[date] = pd.Series(values, index=tickers)
    
    return shuffled


def signal_shuffle_once(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    seed: int
) -> float:
    """
    한 번의 trial: weights 섞기 → 수익률 계산 → Sharpe 계산
    """
    shuffled_weights = shuffle_weights(weights_by_date, seed)
    returns = portfolio_returns_from_weights(prices, shuffled_weights)
    sharpe = calculate_sharpe(returns)
    return sharpe


# ====== Main ======

def main():
    print("="*100)
    print("ML9 SIGNAL SHUFFLE TEST")
    print("="*100)
    
    # 1. 데이터 로딩
    print("\n1. Loading data...")
    prices = load_prices()
    ml9_weights = load_ml9_weights()
    print(f"Prices: {len(prices)} days, {len(prices.columns)} tickers")
    print(f"ML9 weights: {len(ml9_weights)} rebalance dates")
    
    # 2. Baseline (원본 ML9)
    print("\n" + "="*100)
    print("2. BASELINE (Original ML9)")
    print("="*100)
    
    baseline_returns = portfolio_returns_from_weights(prices, ml9_weights)
    baseline_sharpe = calculate_sharpe(baseline_returns)
    print(f"Baseline Sharpe: {baseline_sharpe:.2f}")
    
    # 3. Signal Shuffle Test
    print("\n" + "="*100)
    print("3. SIGNAL SHUFFLE TEST (100 trials)")
    print("="*100)
    
    sharpe_list = []
    for i in range(N_TRIALS):
        seed = 1000 + i
        if (i + 1) % 10 == 0:
            print(f"Trial {i+1}/{N_TRIALS}...")
        
        sh = signal_shuffle_once(prices, ml9_weights, seed)
        sharpe_list.append(sh)
    
    sharpe_arr = np.array(sharpe_list)
    
    # 4. 통계 계산
    print("\n" + "="*100)
    print("4. SHUFFLE STATISTICS")
    print("="*100)
    
    stats = {
        "n_trials": N_TRIALS,
        "baseline_sharpe": float(baseline_sharpe),
        "shuffle_mean_sharpe": float(sharpe_arr.mean()),
        "shuffle_std_sharpe": float(sharpe_arr.std()),
        "shuffle_min_sharpe": float(sharpe_arr.min()),
        "shuffle_max_sharpe": float(sharpe_arr.max()),
        "sharpe_list": sharpe_list,
    }
    
    print(f"Baseline Sharpe: {stats['baseline_sharpe']:.2f}")
    print(f"Shuffle Mean Sharpe: {stats['shuffle_mean_sharpe']:.2f}")
    print(f"Shuffle Std Sharpe: {stats['shuffle_std_sharpe']:.2f}")
    print(f"Shuffle Min Sharpe: {stats['shuffle_min_sharpe']:.2f}")
    print(f"Shuffle Max Sharpe: {stats['shuffle_max_sharpe']:.2f}")
    
    # 5. 결과 저장
    print("\n" + "="*100)
    print("5. SAVING RESULTS")
    print("="*100)
    
    out_path = BASE_DIR / "results" / "ml9_signal_shuffle_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to: {out_path}")
    
    # 6. 해석
    print("\n" + "="*100)
    print("6. INTERPRETATION")
    print("="*100)
    
    if stats['shuffle_mean_sharpe'] < 0.5:
        print("✅ PASS: Shuffle 후 Sharpe가 0 근처로 떨어짐")
        print("   → ML9은 진짜 알파를 가지고 있음")
    elif stats['shuffle_mean_sharpe'] < 1.5:
        print("⚠️ CAUTION: Shuffle 후에도 Sharpe가 양수")
        print("   → 약간의 구조적 바이어스 가능성")
    else:
        print("❌ FAIL: Shuffle 후에도 Sharpe가 높음")
        print("   → 구조적 바이어스가 있을 가능성 높음")
    
    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
