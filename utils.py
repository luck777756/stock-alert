import numpy as np
import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

def calculate_score(df):
    if df.empty or len(df) < 20:
        return 0.0

    price_now = df["Close"].iloc[-1]
    ma20      = df["Close"].rolling(20).mean().iloc[-1]
    price_score = (price_now - ma20) / ma20 if ma20 > 0 else 0.0

    vol_now = df["Volume"].iloc[-1]
    vol_avg = df["Volume"].rolling(5).mean().iloc[-1]
    volume_score = (vol_now - vol_avg) / vol_avg if vol_avg > 0 else 0.0

    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx().iloc[-1]
    adx_score = adx / 100

    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    pb_raw = bb.bollinger_pband().fillna(0)
    arr = pb_raw.values
    if arr.ndim > 1:
        arr = arr.flatten()
    pb_series = pd.Series(arr, index=df.index)
    bb_score = pb_series.iloc[-1] if not pb_series.empty else 0.0

    total = price_score + volume_score + adx_score + bb_score
    return round(total, 4)
