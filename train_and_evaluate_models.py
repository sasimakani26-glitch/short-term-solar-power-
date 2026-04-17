from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
import numpy as np
import pandas as pd
import joblib
import config

# ---------------------------
# Evaluate traditional ML models
# ---------------------------
def evaluate_ml_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append([name, mae, rmse, r2, mape])

    return pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "MAPE"])


# ---------------------------
# Evaluate Deep Learning models (auto-reshape for LSTM/GRU)
# ---------------------------
def evaluate_deep_learning_models(models, X_test, y_test):
    results = []

    # If input is 2D, reshape to 3D for LSTM/GRU: (samples, timesteps=1, features)
    if len(X_test.shape) == 2:
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    for name, model in models.items():
        preds = model.predict(X_test)

        # Flatten predictions if needed
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.flatten()

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mean_absolute_percentage_error = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append([name, mae, rmse, r2, mean_absolute_percentage_error])

    return pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "MAPE"])

def select_best_model(results_df, models):
    best_row = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_row["Model"]

    best_model = models[best_model_name]
    joblib.dump(best_model, config.MODEL_PATH)

    return best_model_name

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# =========================
# Machine Learning Models
# =========================
def train_ml_models(X_train, y_train):
    models = {}
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models

# =========================
# Deep Learning Models
# =========================
def train_deep_learning_models(X_train, y_train):
    models = {}

    # Make sure X_train has shape (samples, timesteps, features)
    X_train_dl = np.array(X_train)
    if len(X_train_dl.shape) == 2:
        # Reshape to (samples, timesteps, features) assuming timesteps=1
        X_train_dl = X_train_dl.reshape((X_train_dl.shape[0], 1, X_train_dl.shape[1]))

    # LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=0)
    models["LSTM"] = lstm_model
    return models