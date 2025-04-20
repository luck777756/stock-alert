import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from ta.volatility import BollingerBands

def load_hist(ticker):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    df.dropna(inplace=True)
    return df

def make_features(df):
    X = pd.DataFrame(index=df.index)
    X['obv'] = ((df['Close'].diff()>0)*df['Volume'] - (df['Close'].diff()<0)*df['Volume']).cumsum()
    X['vol_pct'] = df['Volume'].pct_change().fillna(0)
    X['ma20_diff'] = (df['Close'] - df['Close'].rolling(20).mean())/df['Close'].rolling(20).mean()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    # Bollinger pband with flatten for 2D arrays
    try:
        pb = bb.bollinger_pband()
        if hasattr(pb, 'values') and hasattr(pb, 'ndim') and pb.ndim > 1:
            pb = pb.squeeze()
        pb = pd.Series(pb, index=df.index).fillna(0)
    except Exception as e:
        pb = pd.Series([0] * len(df), index=df.index)
    try:
        pb = pb.squeeze() if hasattr(pb, 'squeeze') else pb
        if pb.ndim > 1:
            pb = pd.Series(pb[:, 0], index=df.index)
    except:
        pb = pd.Series([0]*len(df), index=df.index)
    pb = pb.fillna(0)
    if isinstance(pb, pd.DataFrame):
        pb = pb.iloc[:, 0]
    elif hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.squeeze(), index=df.index)
    pb = pb.fillna(0)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.reshape(-1), index=df.index)
    pb = pb.fillna(0)
    if isinstance(pb.values, np.ndarray) and pb.values.ndim > 1:
        pb = pd.Series(pb.values.reshape(-1), index=df.index)
    pb = pb.fillna(0)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.flatten(), index=df.index)
    pb = pb.fillna(0)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.flatten(), index=df.index)
    pb = pb.fillna(0)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.flatten(), index=df.index)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.flatten(), index=df.index)
    X['bb_pctb'] = pb
    X['adx'] = 0
    return X.dropna()

def label_future(df, days=10, target=0.6):
    fut = df['Close'].shift(-days)
    ret = fut/df['Close'] - 1
    return (ret>=target).astype(int)

if __name__ == '__main__':
    tickers = [t.strip() for t in open("tickers_nasdaq.txt")]
    all_X, all_y = [], []
    for t in tickers:
        df = load_hist(t)
        if len(df)<60: continue
        X = make_features(df)
        y = label_future(df).reindex(X.index).fillna(0).astype(int)
        all_X.append(X); all_y.append(y)
    if all_X:
        X_full = pd.concat(all_X); y_full = pd.concat(all_y)
        tscv = TimeSeriesSplit(n_splits=5)
        params = {'n_estimators':[50,100], 'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
        clf = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                           params, cv=tscv, scoring='precision')
        clf.fit(X_full, y_full)
        joblib.dump(clf.best_estimator_, "best_model.pkl")
        print("Model trained and saved.")