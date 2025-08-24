import pandas as pd
import numpy as np

def clean_data(df):
    selected_features = ['ema_trend_slope', 'rsi', 'macd_histogram', 'atr', 'obv',
                         'sma_ratio_20_50', 'percentile_30', 'Close', 'price_lag5', 'price_lag20', 'price_lag1', 'price_lag50']

    df = df.copy()

    # Drop first rows with NaNs from rolling windows
    df = df.iloc[65:].reset_index()
    df['date'] = df['Date'].dt.date
    df = df.drop(columns=['Date'])

    # Fill NAs on FEATURES ONLY
    df[selected_features] = df[selected_features].ffill().bfill()

    # Build time features AFTER slicing
    df['t_index'] = np.arange(1, len(df) + 1)
    df['t_index_sq'] = df['t_index']**2

    # Target = future raw price
    df['target'] = df['Close'].shift(-30)
    df['target_lstm'] = df['Close'].shift(-30) - df['Close']

    # Drop tail NaNs from the shift
    df = df.dropna(subset=['target']).reset_index(drop=True)
    df = df.dropna(subset=['target_lstm']).reset_index(drop=True)


    keep_cols = ['date', 'target', 'target_lstm', 't_index', 't_index_sq'] + selected_features
    return df[[c for c in keep_cols if c in df.columns]]
