import yfinance as yf
import pandas as pd
import numpy as np

class StockAnalyzer:
    def __init__(self, ticker, start_date, end_date, interval):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stock_data = None

    def download_data(self):
        self.stock_data = yf.download(
            self.ticker, start=self.start_date, end=self.end_date, interval=self.interval
        )
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            self.stock_data.columns = self.stock_data.columns.get_level_values(0)
        self.stock_data['Volume'] = self.stock_data['Volume'].replace(0, np.nan).ffill().bfill()

    def get_stock_data(self):
        return self.stock_data

    # === Feature groups ===

    def add_trend_features(self):
        s = self.stock_data
        s["trend1"] = s["Close"].pct_change(1) * 100
        s["trend3"] = s["Close"].pct_change(3) * 100
        s["trend5"] = s["Close"].pct_change(5) * 100
        s["trend10"] = s["Close"].pct_change(10) * 100
        s["trend20"] = s["Close"].pct_change(20) * 100
        s["trend50"] = s["Close"].pct_change(50) * 100
        s["ema_10"] = s["Close"].ewm(span=10).mean()
        s["ema_50"] = s["Close"].ewm(span=50).mean()
        s["ema_diff"] = s["ema_10"] - s["ema_50"]
        s["ema_10_above_50"] = (s["ema_10"] > s["ema_50"]).astype(int)
        s["log_close"] = np.log(s["Close"])
        s["trend_slope_50"] = s["log_close"].rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        s["ema_trend_slope"] = (s["ema_10"] - s["ema_50"]).rolling(10).mean()
        s['price_lag1']  = s['Close'].shift(1)
        s['price_lag5']  = s['Close'].shift(5)
        s['price_lag20'] = s['Close'].shift(20)
        s['price_lag50'] = s['Close'].shift(50)

    def add_momentum_features(self):
        s = self.stock_data
        delta = s['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        s["rsi"] = 100 - (100 / (1 + rs))
        tp = (s["High"] + s["Low"] + s["Close"]) / 3
        s["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        macd = s["Close"].ewm(span=12).mean() - s["Close"].ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        s["macd_histogram"] = macd - signal
        s["ret_5d_stock"] = s["Close"].pct_change(5)
        s["ret_20d_stock"] = s["Close"].pct_change(20)

    def add_volatility_features(self):
        s = self.stock_data
        s["daily_return"] = s["Close"].pct_change()
        s["volatility_30d"] = s["daily_return"].rolling(30).std()
        s["vol_10d"] = s["daily_return"].rolling(10).std()
        s["vol_50d"] = s["daily_return"].rolling(50).std()
        s["vol_skew"] = s["vol_10d"] / s["vol_50d"]
        s["is_high_vol"] = (s["vol_10d"] > s["vol_10d"].rolling(50).mean()).astype(int)
        hl = s["High"] - s["Low"]
        hc = np.abs(s["High"] - s["Close"].shift())
        lc = np.abs(s["Low"] - s["Close"].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        s["atr"] = tr.rolling(window=14).mean()

    def add_volume_features(self):
        s = self.stock_data
        s["volume_s"] = s["Volume"].ewm(span=5).mean()
        s["volume_l"] = s["Volume"].ewm(span=20).mean()
        s["vol_osi"] = ((s["volume_s"] - s["volume_l"]) / s["volume_l"]) * 100
        s["volume_chg"] = s["Volume"].pct_change()
        s["volume_vol"] = s["volume_chg"].rolling(5).std()
        s["obv"] = (np.sign(s["Close"].diff()) * s["Volume"]).fillna(0).cumsum()

    def add_price_features(self):
        s = self.stock_data
        s["gap"] = (s["Open"] - s["Close"].shift(1)) / s["Close"].shift(1)
        s["gap_rel_range"] = s["gap"] / ((s["High"] - s["Low"]) / s["Close"])
        s["intraday_range"] = (s["High"] - s["Low"]) / s["Close"]
        s["vwap"] = (s["Close"] * s["Volume"]).cumsum() / s["Volume"].cumsum()
        s["vwap_distance"] = s["Close"] - s["vwap"]
        s["cum_vol"] = s["Volume"].cumsum()
        s["cum_vwap"] = (s["Close"] * s["Volume"]).cumsum() / s["cum_vol"]
        s["vwap_dev"] = s["Close"] / s["cum_vwap"] - 1
        rolling_max = s["Close"].rolling(20).max()
        s["drawdown_from_peak"] = s["Close"] / rolling_max - 1

    def add_seasonality_features(self):
        s = self.stock_data
        s["day_of_week"] = s.index.dayofweek / 4.0
        s["up_day"] = s["daily_return"] > 0
        s["breadth_20"] = s["up_day"].rolling(20).mean()
        s.drop(columns=["up_day"], inplace=True)

    def add_ratio_features(self):
        s = self.stock_data
        sma_5 = s["Close"].rolling(5).mean()
        sma_20 = s["Close"].rolling(20).mean()
        sma_50 = s["Close"].rolling(50).mean()
        s["sma_ratio_5_20"] = sma_5 / sma_20
        s["sma_ratio_5_50"] = sma_5 / sma_50
        s["sma_ratio_20_50"] = sma_20 / sma_50
    
    def add_altitude_features(self):
        s = self.stock_data

        # 2. Z-score of price over 30-day window
        s["zscore_30"] = (s["Close"] - s["Close"].rolling(50).mean()) / s["Close"].rolling(50).std()

        # 3. Percentile rank of price over 30-day window
        s["percentile_30"] = s["Close"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        # 4. Normalized price position in recent 30-day range
        rolling_min = s["Close"].rolling(50).min()
        rolling_max = s["Close"].rolling(50).max()
        s["norm_price_30"] = (s["Close"] - rolling_min) / (rolling_max - rolling_min)

        # 5. Cumulative return since start
        s["cum_return"] = s["Close"] / s["Close"].iloc[0] - 1


    def analyze_stock(self):
        self.download_data()
        self.add_trend_features()
        self.add_momentum_features()
        self.add_volatility_features()
        self.add_volume_features()
        self.add_price_features()
        self.add_seasonality_features()
        self.add_ratio_features()
        self.add_altitude_features()
