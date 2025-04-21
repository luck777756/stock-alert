import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBV, 거래량 증감률, MA20 차이, Bollinger %B, ADX 특성 생성
    """
    X = pd.DataFrame(index=df.index)
    diffs = df['Close'].diff().fillna(0)
    X['obv'] = (diffs.gt(0) * df['Volume'] - diffs.lt(0) * df['Volume']).cumsum()
    X['vol_pct'] = df['Volume'].pct_change().fillna(0)
    X['ma20_diff'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()

    # Bollinger Bands %B
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    pb = bb.bollinger_pband().fillna(0).values
    arr = pb.flatten() if pb.ndim > 1 else pb
    X['bb_pctb'] = pd.Series(arr, index=df.index)

    # ADX (14)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx().fillna(0)
    X['adx'] = adx

    return X.dropna()

def calculate_score(df: pd.DataFrame) -> float:
    """
    price, volume, ADX, Bollinger %B 융합 점수 계산
    """
    if df.empty or len(df) < 20:
        return 0.0

    price_now = df['Close'].iat[-1]
    ma20 = df['Close'].rolling(20).mean().iat[-1]
    price_score = (price_now - ma20) / ma20 if ma20 else 0.0

    vol_now = df['Volume'].iat[-1]
    vol_avg = df['Volume'].rolling(5).mean().iat[-1]
    volume_score = (vol_now - vol_avg) / vol_avg if vol_avg else 0.0

    adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx().iat[-1]
    adx_score = adx / 100

    pb = BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_pband().fillna(0).values
    arr = pb.flatten() if pb.ndim > 1 else pb
    bb_score = arr[-1] if len(arr) else 0.0

    return round(price_score + volume_score + adx_score + bb_score, 4)
