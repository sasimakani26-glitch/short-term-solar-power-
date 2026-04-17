from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config
import pandas as pd
import json
import os


def load_dataset(path):
    return pd.read_csv(path)

def validate_data(df):
    return df.dropna()


def split_and_scale(df):
    # handle timestamp column: convert and extract numeric features
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Hour"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        df["Month"] = df["Timestamp"].dt.month
        df = df.drop(columns=["Timestamp"])
    # create lag features from Power_Output (previous timesteps)
    if "Power_Output" in df.columns and hasattr(config, "LAG_FEATURES"):
        max_lag = max(config.LAG_FEATURES) if len(config.LAG_FEATURES) > 0 else 0
        for lag in config.LAG_FEATURES:
            df[f"Power_lag_{lag}"] = df["Power_Output"].shift(lag)
        # drop rows with NaNs introduced by shifting
        if max_lag > 0:
            df = df.dropna().reset_index(drop=True)

    X = df.drop("Power_Output", axis=1)
    # save feature column names so the serving app can construct matching input
    try:
        cols = X.columns.tolist()
        os.makedirs(os.path.dirname(config.FEATURE_COLUMNS_PATH), exist_ok=True)
        with open(config.FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as fh:
            json.dump(cols, fh)
    except Exception:
        pass
    y = df["Power_Output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
