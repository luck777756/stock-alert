import numpy as np
import pandas as pd
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
    if isinstance(pb, pd.DataFrame) or (hasattr(pb, 'values') and isinstance(pb.values, np.ndarray) and pb.values.ndim > 1):
    pb = pd.Series(pb.values.flatten())
    pb = pb.fillna(0)
    bb_score = pb.iloc[-1] if not pb.empty else 0
    return round((price_now - ma20) / ma20 + volume_score + adx / 100 + bb_score, 4)
