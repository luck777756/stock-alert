import os, time, logging
import datetime, requests, joblib, pandas as pd, yfinance as yf

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.volatility import BollingerBands

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
NTFY = os.getenv('NTFY_CHANNEL','https://ntfy.sh/my-stock-alert')
SCAN_INTERVAL, SEND_INTERVAL, URGENT_INTERVAL = 60, 3600, 1200
STRICT_CAP, PRICE_LIMIT = (300_000_000,1_500_000_000), 10
TICKERS_FILE = 'tickers_nasdaq.txt'; DELISTED={'TMBR'}



# Load ML model if available
try:
    model = joblib.load('best_model.pkl')
    logging.info('ML model loaded.')
except Exception:
    model = None
    logging.warning('No ML model, skipping ML filter.')

analyzer = SentimentIntensityAnalyzer()

def send_msg(text, stage, urgent=False):
    if not hasattr(send_msg,'last'):
        send_msg.last = {}
    now = time.time()
    interval = URGENT_INTERVAL if urgent else SEND_INTERVAL
    if stage in send_msg.last and now - send_msg.last[stage] < interval:
        return
    headers = {'Priority':'urgent'} if urgent else {}
    try:
        requests.post(NTFY, data=text.encode(), headers=headers)
        send_msg.last[stage] = now
    except Exception as e:
        logging.error(f"send_msg error: {e}")

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period='1mo', interval='1d', auto_adjust=True)
        df.dropna(inplace=True)
        info = yf.Ticker(ticker).info
        return df, info
    except Exception as e:
        logging.error(f"fetch_data error for {ticker}: {e}")
        return None, None

def make_features(df):
    X = pd.DataFrame(index=df.index)
    diffs = df['Close'].diff().fillna(0)
    X['obv'] = (diffs.gt(0) * df['Volume'] - diffs.lt(0) * df['Volume']).cumsum()
    X['vol_pct'] = df['Volume'].pct_change().fillna(0)
    X['ma20_diff'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    pb = bb.bollinger_pband().fillna(0)
    if hasattr(pb, 'values') and pb.values.ndim > 1:
        pb = pd.Series(pb.values.flatten(), index=df.index)
    X['bb_pctb'] = pb
    X['adx'] = 0
    return X.dropna()

def determine_strategy(df):
    entry = round(df['Close'].iat[-1], 2)
    target = round(entry * 1.5, 2)
    session = '프장 진입' if df['High'].iat[-1] - df['Low'].iat[-1] > 0.5 else '본장 진입'
    return entry, target, session

def format_alert(prefix, ticker, info, grade, rank, entry, target, session, basis):
    name = info.get('shortName','')
    cap = f"${info.get('marketCap',0)//1_000_000:,}M"
    return (
        f"{prefix}\n"
        f"🚀 [{ticker}] 매집 돌파 감지!\n"
        f"📈 이름: {name}\n"
        f"🏅 랭킹: {rank}위 ({grade}등급)\n"
        f"💸 진입가: ${entry} / 매도가: ${target}\n"
        f"🕰️ 시점 분석: {session}\n"
        f"📊 분석 기반: {basis}\n"
    )

def calculate_score(df):
    c, v = df['Close'], df['Volume']
    diffs = c.diff().fillna(0)
    obv = (diffs.gt(0) * v - diffs.lt(0) * v).cumsum()
    return (
        (obv.iat[-1] - obv.iat[-5]) * 0.5
        + ((v.iat[-1] - v.iat[-2]) / v.iat[-2] if v.iat[-2] > 0 else 0) * 0.3
        + ((c.iat[-1] - c.rolling(20).mean().iat[-1]) / c.rolling(20).mean().iat[-1]) * 0.2
    )

def main_loop():
    tickers = [t.strip() for t in open(TICKERS_FILE)]
    # Strict stage
    results = []
    for t in tickers:
        if t in DELISTED: continue
        df, info = fetch_data(t)
        if df is None or df.empty or len(df)<21 or 'marketCap' not in info: continue
        if not (STRICT_CAP[0] <= info['marketCap'] <= STRICT_CAP[1]) or df['Close'].iat[-1]>PRICE_LIMIT: continue
        score = calculate_score(df)
        grade = '1' if score>50 else '1.5' if score>20 else '2' if score>5 else '3'
        results.append((t, info, grade, score, df))
    strict = sorted([r for r in results if r[2] in ['1','1.5']], key=lambda x:x[3], reverse=True)[:5]
    if strict:
        prefix = "✅ Strict 상위 5개:"
        for i,(t,info,grade,score,df) in enumerate(strict,1):
            entry,target,session = determine_strategy(df)
            basis="OBV↑, Bollinger%, MA20 diff"
            msg = format_alert(prefix, t, info, grade, i, entry, target, session, basis)
            send_msg(msg,'strict', urgent=True)
        return
    logging.info("Strict candidates: 0, moving to Fallback stage")
    # Fallback stage
    results = []
    for t in tickers:
        df, info = fetch_data(t)
        if df is None or df.empty or len(df)<21 or 'marketCap' not in info: continue
        score = calculate_score(df)
        grade = '1' if score>50 else '1.5' if score>20 else '2' if score>5 else '3'
        results.append((t, info, grade, score, df))
    fallback = sorted(results, key=lambda x: x[3], reverse=True)[:5]
    if fallback:
        prefix = "⚡ Fallback 상위 5개:"
        for i,(t,info,grade,score,df) in enumerate(fallback,1):
            entry,target,session = determine_strategy(df)
            basis="OBV↑, Bollinger%, MA20 diff"
            msg = format_alert(prefix, t, info, grade, i, entry, target, session, basis)
            send_msg(msg,'fallback', urgent=False)

if __name__=='__main__':
    sent_watch = False
    logging.info("✅ 시스템 v50 시작")
    send_msg("✅ 시스템 v50 시작","startup", urgent=False)
    while True:
        now = datetime.datetime.now()
        if should_send_watchlist() and not sent_watch:
            daily_watchlist()
            sent_watch = True
        main_loop()
        time.sleep(SCAN_INTERVAL)
def daily_watchlist():
    tickers = [t.strip() for t in open(TICKERS_FILE)]
    results = []
    for t in tickers:
        df, info = fetch_data(t)
        if df is None or df.empty or len(df)<21 or 'marketCap' not in info: continue
        score = calculate_score(df)
        grade = '1' if score > 50 else '1.5' if score > 20 else '2' if score > 5 else '3'
        results.append((t, info, grade, score, df))
    top = sorted(results, key=lambda x: x[3], reverse=True)[:5]
    if top:
        prefix = "💤 예비 종목 알림:"
        for i,(t,info,grade,score,df) in enumerate(top,1):
            entry,target,session = determine_strategy(df)
            basis="OBV↑, Bollinger%, MA20 diff"
            msg = format_alert(prefix, t, info, grade, i, entry, target, session, basis)
            send_msg(msg,'daily_watch', urgent=False)

def should_send_watchlist():
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # Saturday, Sunday
        return True
    if now.hour >= 17:  # Assume 5PM local time is after US market close
        return True
    return False
