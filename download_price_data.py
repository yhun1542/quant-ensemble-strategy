#!/usr/bin/env python3
"""
30종목 메가캡 가격 데이터 다운로드 (Yahoo Finance API 사용)
"""
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path

# 30종목 메가캡 유니버스
UNIVERSE_30 = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
]

def download_stock_data(symbol, start_date="2015-01-01", end_date="2024-12-31"):
    """
    특정 종목의 일간 가격 데이터 다운로드
    """
    client = ApiClient()
    
    try:
        print(f"  Downloading {symbol}...", end=" ")
        
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': symbol,
            'region': 'US',
            'interval': '1d',
            'range': 'max',  # 최대 기간
            'includeAdjustedClose': True
        })
        
        if response and 'chart' in response and 'result' in response['chart']:
            result = response['chart']['result'][0]
            
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # DataFrame 생성
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # 날짜 필터링
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = df.set_index('date').sort_index()
            
            # NaN 제거
            df = df.dropna()
            
            print(f"✓ ({len(df)} days)")
            return df['close']
            
        else:
            print(f"✗ No data")
            return None
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def main():
    print("=" * 80)
    print("30종목 메가캡 가격 데이터 다운로드")
    print("=" * 80)
    
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    print(f"\n기간: {start_date} ~ {end_date}")
    print(f"종목 수: {len(UNIVERSE_30)}\n")
    
    # 각 종목별 데이터 다운로드
    price_data = {}
    
    for symbol in UNIVERSE_30:
        prices = download_stock_data(symbol, start_date, end_date)
        
        if prices is not None and len(prices) > 0:
            price_data[symbol] = prices
        
        # API 호출 제한 방지
        time.sleep(0.5)
    
    # DataFrame으로 결합
    if price_data:
        df_prices = pd.DataFrame(price_data)
        
        print(f"\n결과:")
        print(f"  - 성공적으로 다운로드된 종목 수: {len(df_prices.columns)}")
        print(f"  - 총 거래일 수: {len(df_prices)}")
        print(f"  - 기간: {df_prices.index[0].date()} ~ {df_prices.index[-1].date()}")
        print(f"  - 결측치 비율: {df_prices.isna().sum().sum() / (df_prices.shape[0] * df_prices.shape[1]) * 100:.2f}%")
        
        # 데이터 디렉토리 생성
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Parquet 형식으로 저장
        output_path = data_dir / "price_data_sp500.parquet"
        df_prices.to_parquet(output_path)
        
        print(f"\n✅ 저장 완료: {output_path}")
        
        # 샘플 데이터 출력
        print(f"\n샘플 데이터 (최근 5일):")
        print(df_prices.tail(5))
        
    else:
        print("\n❌ 다운로드된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
