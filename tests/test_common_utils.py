import pandas as pd
from utils.common_utils import make_features, calculate_score

def test_make_features_basic():
    dates = pd.date_range('2021-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Open': range(30),
        'High': range(1,31),
        'Low': range(0,30),
        'Close': range(30),
        'Volume': [100]*30
    }, index=dates)
    X = make_features(df)
    assert 'obv' in X.columns
    assert len(X) > 0

def test_calculate_score_zero():
    df_empty = pd.DataFrame()
    assert calculate_score(df_empty) == 0.0
