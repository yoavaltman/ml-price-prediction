import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def make_sequences(X, y, window_size):
    Xs, ys, idx = [], [], []
    for i in range(len(X) - window_size + 1):
        start = i
        end = i + window_size 
        Xs.append(X.iloc[start:end].values)
        ys.append(y.iloc[end - 1])
        idx.append(X.index[end - 1])
    return np.array(Xs), np.array(ys), np.array(idx)


def train_lstm_model(X_train, y_train, X_val, y_val, best_epoch=None, window_size=60,
                     units1=128, units2=128, learning_rate=0.001, batch_size=64, epochs=100):

    # Create sequences
    X_seq_train, y_seq_train, _ = make_sequences(X_train, y_train, window_size)
    X_seq_val,   y_seq_val,   _ = make_sequences(X_val,   y_val,   window_size)

    if len(X_seq_train) == 0 or len(X_seq_val) == 0:
        raise ValueError("Not enough data to form sequences with current window size.")

    input_shape = X_seq_train.shape[1:]

    # Model
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units1, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences = True),
        Dropout(0.2),
        LSTM(128, return_sequences = True),
        Dropout(0.2),
        LSTM(units2),
        Dropout(0.2),
        Dense(1, activation='linear', kernel_initializer='he_uniform')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=["mae"])

    if best_epoch is None:
        # Phase 1: tune using validation
        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        history = model.fit(X_seq_train, y_seq_train,
                            validation_data=(X_seq_val, y_seq_val),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stop], verbose=0)
        return model, history, window_size
    else:
        # Phase 2: retrain on combined train+val with best_epoch
        # Rebuild sequences on concatenated raw data to include cross-boundary windows
        X_fit = pd.concat([X_train, X_val], axis=0).sort_index()
        y_fit = pd.concat([y_train, y_val], axis=0).sort_index()
        X_seq_full, y_seq_full, _ = make_sequences(X_fit, y_fit, window_size)

        history = model.fit(X_seq_full, y_seq_full,
                            epochs=best_epoch, batch_size=batch_size, verbose=0)
        return model, history, window_size



def get_best_epoch(history):
    """
    Extract the best epoch (1-based) based on minimum validation loss.
    """
    if 'val_loss' not in history.history:
        raise RuntimeError("History missing 'val_loss'. Ensure validation_data was provided.")
    return int(np.argmin(history.history['val_loss'])) + 1  # 1-based


def train_linear_model(X_train, y_train, X_val, y_val):
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)

    model = LinearRegression()
    model.fit(X_fit, y_fit)
    return model
 

def model_predict(model, X, y, is_lstm=False):
    if is_lstm:
        preds = model.predict(X, verbose=0).flatten()
    else:
        preds = model.predict(X).flatten()

    actuals = np.asarray(y).flatten()
    return model, preds, actuals

def evaluate_model(price_preds, price_actual, label=""):
    mse = mean_squared_error(price_actual, price_preds)
    r2 = r2_score(price_actual, price_preds)
    print(f"{label} Price MSE: {mse:.6f}")
    print(f"{label} Price R²: {r2:.4f}")
    return mse, r2
