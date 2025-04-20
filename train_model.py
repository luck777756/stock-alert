
import pandas as pd
import numpy as np
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

def make_features(df):
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    pb = bb.bollinger_pband()
    if isinstance(pb.values, pd.DataFrame) or (hasattr(pb.values, 'ndim') and pb.values.ndim > 1):
        pb = pd.Series(pb.values.flatten())
    df["bb_pctb"] = pb.fillna(0)
    df["adx"] = ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx().fillna(0)
    df["return"] = df["Close"].pct_change().fillna(0)
    df["target"] = (df["Close"].shift(-3) > df["Close"] * 1.3).astype(int)
    return df.dropna()

ticker = "APCX"
df = yf.download(ticker, period="6mo", interval="1d")
df = make_features(df)

X = df[["bb_pctb", "adx", "return"]]
y = df["target"]

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=tscv)
clf.fit(X, y)

joblib.dump(clf.best_estimator_, "best_model.pkl")
