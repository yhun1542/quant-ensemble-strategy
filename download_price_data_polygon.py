#!/usr/bin/env python3
"""
30종목 메가캡 가격 데이터 다운로드 (Polygon API 사용)
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path

# Polygon API 설정
POLYGON_API_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"

# 30종목 메가캡 유니버스
UNIVERSE_30 = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
]

def download_stock_data_polygon(symbol, start_date="2015-01-01", end_date="2024-12-31"):
    """
    Polygon API를 사용하여 특정 종목의 일간 가격 데이터 다운로드
    """
    try:
        print(f"  Downloading {symbol}...", end=" ", flush=True)
        
        # Polygon API endpoint
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                results = data["results"]
                
                # DataFrame 생성
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.set_index('date').sort_index()
                
                # NaN 제거
                df = df.dropna()
                
                print(f"✓ ({len(df)} days)")
                return df['close']
            else:
                print(f"✗ No data")
                return None
        else:
            print(f"✗ Error {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def main():
    print("=" * 80)
    print("30종목 메가캡 가격 데이터 다운로드 (Polygon API)")
    print("=" * 80)
    
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    print(f"\n기간: {start_date} ~ {end_date}")
    print(f"종목 수: {len(UNIVERSE_30)}\n")
    
    # 각 종목별 데이터 다운로드
    price_data = {}
    
    for i, symbol in enumerate(UNIVERSE_30, 1):
        prices = download_stock_data_polygon(symbol, start_date, end_date)
        
        if prices is not None and len(prices) > 0:
            price_data[symbol] = prices
        
        # API 호출 제한 방지 (Polygon free tier: 5 calls/min)
        if i % 5 == 0 and i < len(UNIVERSE_30):
            print(f"  (API 제한 방지를 위해 60초 대기...)")
            time.sleep(60)
        else:
            time.sleep(1)
    
    # DataFrame으로 결합
    if price_data:
        df_prices = pd.DataFrame(price_data)
        
        # 모든 종목이 존재하는 날짜만 유지 (결측치 제거)
        df_prices = df_prices.dropna()
        
        print(f"\n결과:")
        print(f"  - 성공적으로 다운로드된 종목 수: {len(df_prices.columns)}")
        print(f"  - 총 거래일 수: {len(df_prices)}")
        print(f"  - 기간: {df_prices.index[0].date()} ~ {df_prices.index[-1].date()}")
        print(f"  - 결측치: {df_prices.isna().sum().sum()} (제거 완료)")
        
        # 데이터 디렉토리 생성
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # CSV 형식으로 저장
        output_path = data_dir / "price_data_sp500.csv"
        df_prices.to_csv(output_path)
        
        print(f"\n✅ 저장 완료: {output_path}")
        
        # 샘플 데이터 출력
        print(f"\n샘플 데이터 (최근 5일):")
        print(df_prices.tail(5).round(2))
        
        # 통계 정보
        print(f"\n가격 통계:")
        print(df_prices.describe().round(2))
        
    else:
        print("\n❌ 다운로드된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
