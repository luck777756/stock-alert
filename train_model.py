import os, time, random, logging
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from ta.volatility import BollingerBands
import shutil

# 로그 설정
d_logging = logging.basicConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 캐시 디렉토리\CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_hist(ticker, base_sleep=0.1, max_retry=3):
    """
    Adaptive backoff과 로컬 캐시를 활용하여 Yahoo Finance 데이터 로드
    캐시 파일이 있으면 우선 로드하고, 없으면 다운로드 시도
    """
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    # 캐시 로드
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logging.info(f"{ticker}: 캐시 로드 성공")
            return df
        except Exception as e:
            logging.error(f"{ticker}: 캐시 로드 실패: {e}")
    
    # adaptive backoff retry
    for i in range(max_retry):
        try:
            df = yf.download(
                ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                threads=True,
                group_by='ticker'
            )
            df.dropna(inplace=True)
            if not df.empty:
                df.to_csv(path)
                logging.info(f"{ticker}: 다운로드 및 캐시 저장 성공")
                return df
        except Exception as e:
            logging.error(f"{ticker}: 다운로드 실패 ({i+1}/{max_retry}): {e}")
        time.sleep(base_sleep * (2**i) + random.random()*0.1)
    logging.error(f"{ticker}: 다운로드 최종 실패 after {max_retry} retries")
    return None


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X['obv'] = ((df['Close'].diff() > 0) * df['Volume'] - (df['Close'].diff() < 0) * df['Volume']).cumsum()
    X['vol_pct'] = df['Volume'].pct_change().fillna(0)
    X['ma20_diff'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    pb = bb.bollinger_pband().fillna(0)
    arr = pb.values
    if arr.ndim > 1:
        arr = arr.flatten()
    X['bb_pctb'] = pd.Series(arr, index=df.index)
    X['adx'] = 0
    return X.dropna()


def label_future(df, days=10, target=0.6):
    fut = df['Close'].shift(-days)
    ret = fut / df['Close'] - 1
    labels = (ret >= target).astype(int)
    return labels.dropna()


if __name__ == '__main__':
    # 티커 리스트 로드
    try:
        with open("tickers_nasdaq.txt") as f:
            tickers = [t.strip() for t in f if t.strip()]
    except FileNotFoundError:
        logging.error("tickers_nasdaq.txt 파일을 찾을 수 없습니다.")
        exit(1)

    all_X, all_y = [], []
    for t in tickers:
        df = load_hist(t)
        if df is None or len(df) < 60:
            continue
        X = make_features(df)
        y = label_future(df, days=10, target=0.6)
        # 피처-레이블 일치하도록 교집합 인덱스 사용
        idx = X.index.intersection(y.index)
        all_X.append(X.loc[idx])
        all_y.append(y.loc[idx])

    if not all_X:
        logging.error("유효한 학습 데이터가 없습니다.")
        exit(1)

    # 데이터 병합 및 정렬
    X_full = pd.concat(all_X).sort_index()
    y_full = pd.concat(all_y).sort_index()

    # 모델 학습
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = {'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
    params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    clf = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        params,
        cv=tscv,
        scoring=scoring,
        refit='precision',
        return_train_score=False,
        n_jobs=-1
    )
    clf.fit(X_full, y_full)
    logging.info(f"Best Precision: {clf.best_score_:.4f} with params {clf.best_params_}")

    # 모델 저장 및 압축
    model_path = "best_model.pkl"
    joblib.dump(clf.best_estimator_, model_path)
    shutil.make_archive("trained_model", 'zip', '.', model_path)
    logging.info("📦 모델 학습 완료 및 trained_model.zip 생성됨")
