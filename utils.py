
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

def calculate_score(df):
    if df.empty or len(df) < 20:
        return 0.0
    price_now = df["Close"].iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    vol_now = df["Volume"].iloc[-1]
    vol_avg = df["Volume"].rolling(5).mean().iloc[-1]
    volume_score = (vol_now - vol_avg) / vol_avg if vol_avg > 0 else 0

    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx().iloc[-1]
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    pb = bb.bollinger_pband()
    if isinstance(pb.values, pd.DataFrame) or (hasattr(pb.values, 'ndim') and pb.values.ndim > 1):
        pb = pd.Series(pb.values.flatten())
    bb_score = pb.fillna(0).iloc[-1]

    return round((price_now - ma20) / ma20 + volume_score + adx / 100 + bb_score, 4)
