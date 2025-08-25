import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from StockDataAnalyzer1 import StockAnalyzer
from prepare_data1 import clean_data
from split_data1 import split_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

from model_train1 import (
    make_sequences,
    train_lstm_model, get_best_epoch,
    train_linear_model,
    model_predict,
    evaluate_model
)

# ----------------------------
# helpers
# ----------------------------
def save_metrics_row(model_name, dataset, mse, r2, filename="results/metrics1.csv"):
    row = {"model": model_name, "dataset": dataset, "mse": mse, "r2_score": r2}
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def save_predictions_csv(dates, actual_price, pred_price, actual_log=None, pred_log=None, model_label="", filename="results/preds.csv"):
    dates = pd.Series(dates).reset_index(drop=True)
    actual_price = np.asarray(actual_price).flatten()
    pred_price = np.asarray(pred_price).flatten()
    n = min(len(dates), len(actual_price), len(pred_price))
    df = pd.DataFrame({
        "date": dates.iloc[:n].values,
        "actual_price": actual_price[:n],
        "predicted_price": pred_price[:n],
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {model_label} predictions to {filename}")


def plot_all(dates, actual, lstm_preds, ar_preds, linear_preds, label="Train+Val", save_path=None):
    dates = pd.Series(dates)
    actual = np.asarray(actual).flatten()
    lstm_preds = np.asarray(lstm_preds).flatten()
    ar_preds = np.asarray(ar_preds).flatten()
    linear_preds = np.asarray(linear_preds).flatten()
    n = min(len(dates), len(actual), len(lstm_preds), len(ar_preds), len(linear_preds))
    dates = dates.iloc[:n]; actual = actual[:n]; lstm_preds = lstm_preds[:n]; ar_preds = ar_preds[:n]; linear_preds = linear_preds[:n]

    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual Price")
    plt.plot(dates, lstm_preds, label="LSTM")
    plt.plot(dates, ar_preds, label="AR")
    plt.plot(dates, linear_preds, label="Linear Regression")
    plt.xlabel("Date"); plt.ylabel("Close Price")
    plt.title(f"Actual vs Predicted ({label})")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# ----------------------------
# data loading / prep
# ----------------------------
def load_and_prepare_data():
    analyzer = StockAnalyzer('SPY', '1998-01-01', '2025-07-09', '1d')
    analyzer.analyze_stock()
    df = analyzer.get_stock_data()
    df_clean = clean_data(df)

    # Split
    df_train, df_val, df_test = split_dataframe(df_clean)

    # Dates
    test_dates = df_test['date'].reset_index(drop=True)
    trainval_dates = pd.concat([df_train['date'], df_val['date']]).reset_index(drop=True)

    # Close levels (used for reconstruction/plots)
    test_close = df_test['Close'].reset_index(drop=True)
    trainval_close = pd.concat([df_train['Close'], df_val['Close']]).reset_index(drop=True)

    # Feature set = all except target/date/Close (Close excluded as a feature per your current setup)
    feature_cols = df_clean.columns.difference(['date', 'target', 'target_lstm', 'Close'])

    # "time-only" features for the simple linear model
    X_t = df_train[['t_index', 't_index_sq']].reset_index(drop=True)
    X_v = df_val[['t_index', 't_index_sq']].reset_index(drop=True)
    X   = df_test[['t_index', 't_index_sq']].reset_index(drop=True)

    # Scale features (fit on train only)
    scaler_X = StandardScaler()
    df_train[feature_cols] = scaler_X.fit_transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler_X.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler_X.transform(df_test[feature_cols])

    # Build X/y per split with reset indices
    X_train = df_train[feature_cols].reset_index(drop=True)
    X_val   = df_val[feature_cols].reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)

    # Targets:
    # y_*:     your "level" target (e.g., price level or log price level depending on your pipeline)
    # y_*_lstm: your LSTM change target (Δ = P_{t+30} - P_t or log-return), trained on for stationarity
    y_train = df_train['target'].reset_index(drop=True)
    y_val   = df_val['target'].reset_index(drop=True)
    y_test  = df_test['target'].reset_index(drop=True)

    y_train_lstm = df_train['target_lstm'].reset_index(drop=True)
    y_val_lstm   = df_val['target_lstm'].reset_index(drop=True)
    y_test_lstm  = df_test['target_lstm'].reset_index(drop=True)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), \
           y_train_lstm, y_val_lstm, y_test_lstm, \
           test_dates, trainval_dates, test_close, trainval_close, \
           X_t, X_v, X

# ----------------------------
# main pipeline
# ----------------------------
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test,
                              y_train_lstm, y_val_lstm, y_test_lstm,
                              test_dates, trainval_dates, test_close, trainval_close,
                              X_t, X_v, X):
    os.makedirs("results/plots1", exist_ok=True)
    os.makedirs("models1", exist_ok=True)

    model_path = "models1/lstm_model1.h5"

    # ---------------- LSTM (tune + refit or load) ----------------
    if os.path.exists(model_path):
        print(f"Loading existing LSTM model from {model_path}...")
        lstm_refit = load_model(model_path, compile=False)
        window_size = 60  # TODO: if different, load from a sidecar file
    else:
        print("\n[1/6] Training LSTM (tune on val for early stopping)...")
        lstm_tuned, history, window_size = train_lstm_model(X_train, y_train_lstm, X_val, y_val_lstm)
        best_epoch = get_best_epoch(history)
        print(f"Best epoch (from val_loss): {best_epoch}")

        print("\n[2/6] Re-training LSTM on train+val for best_epoch...")
        lstm_refit, _, _ = train_lstm_model(
            X_train, y_train_lstm, X_val, y_val_lstm,
            best_epoch=best_epoch, window_size=window_size
        )
        lstm_refit.save(model_path)

    # ---------------- Linear baselines ----------------
    print("\n[3/6] Training Autoregression-type (features) ...")
    ar_model = train_linear_model(X_train, y_train, X_val, y_val)
    joblib.dump(ar_model, "models1/AR_model1.pkl")

    print("Training Linear Regression (time-only features)...")
    linear_model = train_linear_model(X_t, y_train, X_v, y_val)
    joblib.dump(linear_model, "models1/linear_model1.pkl")

    # ---------------- Fit sets (train+val) ----------------
    X_fit = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_fit = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    y_fit_lstm = pd.concat([y_train_lstm, y_val_lstm], axis=0).reset_index(drop=True)
    X_time_fit = pd.concat([X_t, X_v], axis=0).reset_index(drop=True)

    # ---- LSTM: sequences & predict (NOTE: y_seq_fit used only to align indices) ----
    X_seq_fit, y_seq_fit, idx_fit = make_sequences(X_fit, y_fit, window_size)

    # Predict change with the refit LSTM
    _, preds_lstm_fit_change, actual_fit_level = model_predict(lstm_refit, X_seq_fit, y_seq_fit, is_lstm=True)

    # Reconstruct level by adding the correct base Close aligned to idx_fit
    base_close_trainval = trainval_close.iloc[idx_fit].to_numpy().reshape(-1)
    preds_lstm_fit_level = preds_lstm_fit_change.reshape(-1) + base_close_trainval
    actual_fit_level = actual_fit_level.reshape(-1)  # already aligned to idx_fit by model_predict

    # ---- AR / Linear: predict on full, then slice to idx_fit for fair comparison ----
    _, preds_ar_full_fit, actual_ar_full_fit = model_predict(ar_model, X_fit, y_fit, is_lstm=False)
    _, preds_linear_full_fit, _ = model_predict(linear_model, X_time_fit, y_fit, is_lstm=False)
    preds_ar_fit = preds_ar_full_fit[idx_fit]
    preds_linear_fit = preds_linear_full_fit[idx_fit]

    # Dates aligned to the same rows as LSTM targets
    dates_trainval_aligned = trainval_dates.iloc[idx_fit].reset_index(drop=True)

    # 5) Plot train+val comparison (saved)
    plot_all(dates_trainval_aligned, actual_fit_level, preds_lstm_fit_level, preds_ar_fit, preds_linear_fit,
             label="Train+Val", save_path="results/plots1/trainval_comparison.png")

    # ----------------------------
    # 6) Test evaluation
    # ----------------------------
    print("\n[6/6] Evaluating on Test Set...")

    # ---- LSTM test ----
    X_seq_test, y_seq_test, idx_test = make_sequences(X_test, y_test, window_size)
    _, preds_lstm_test_change, actual_test_level = model_predict(lstm_refit, X_seq_test, y_seq_test, is_lstm=True)

    base_close_test = test_close.iloc[idx_test].to_numpy().reshape(-1)
    preds_lstm_test_level = preds_lstm_test_change.reshape(-1) + base_close_test
    actual_test_level = actual_test_level.reshape(-1)

    mse_lstm, r2_lstm = evaluate_model(preds_lstm_test_level, actual_test_level, label="LSTM (Test)")
    save_metrics_row("LSTM", "Test", mse_lstm, r2_lstm)
    dates_test_aligned = test_dates.iloc[idx_test].reset_index(drop=True)
    save_predictions_csv(dates_test_aligned, actual_test_level, preds_lstm_test_level,
                         None, None, "LSTM-Test", "results/lstm_preds_test.csv")

    # ---- AR test ----
    _, preds_ar_test_full, actual_ar_full = model_predict(ar_model, X_test, y_test, is_lstm=False)
    mse_ar, r2_ar = evaluate_model(preds_ar_test_full, actual_ar_full, label="AR (Test)")
    save_metrics_row("AR", "Test", mse_ar, r2_ar)
    save_predictions_csv(test_dates, actual_ar_full, preds_ar_test_full,
                         None, None, "AR-Test", "results/ar_preds_test.csv")

    # ---- Linear test ----
    _, preds_lin_test_full, actual_lin_full = model_predict(linear_model, X, y_test, is_lstm=False)
    mse_lin, r2_lin = evaluate_model(preds_lin_test_full, actual_lin_full, label="Linear (Test)")
    save_metrics_row("Linear", "Test", mse_lin, r2_lin)
    save_predictions_csv(test_dates, actual_lin_full, preds_lin_test_full,
                         None, None, "Linear-Test", "results/linear_preds_test.csv")

    # ---- Plot test comparison (aligned to LSTM indices) ----
    preds_ar_test = preds_ar_test_full[idx_test]
    preds_lin_test = preds_lin_test_full[idx_test]
    plot_all(dates_test_aligned, actual_test_level, preds_lstm_test_level, preds_ar_test, preds_lin_test,
             label="Test", save_path="results/plots1/test_comparison.png")

if __name__ == "__main__":
    np.random.seed(42)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
    y_train_lstm, y_val_lstm, y_test_lstm, \
    test_dates, trainval_dates, test_close, trainval_close, \
    X_t, X_v, X = load_and_prepare_data()

    train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        y_train_lstm, y_val_lstm, y_test_lstm,
        test_dates, trainval_dates, test_close, trainval_close,
        X_t, X_v, X
    )
