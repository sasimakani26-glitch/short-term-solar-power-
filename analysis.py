import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# BASIC METRICS
# ==========================================================
def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    # R2 Score
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }


# ==========================================================
# WEATHER ANALYSIS (CLOUDY vs SUNNY)
# ==========================================================
def evaluate_weather_performance(df, y_true, y_pred):

    df = df.copy()
    df["Actual"] = y_true
    df["Predicted"] = y_pred

    sunny = df[df["Cloud_Cover"] < 30]
    cloudy = df[df["Cloud_Cover"] > 70]

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    return {
        "Sunny_RMSE": rmse(sunny["Actual"], sunny["Predicted"]) if len(sunny) > 0 else None,
        "Cloudy_RMSE": rmse(cloudy["Actual"], cloudy["Predicted"]) if len(cloudy) > 0 else None,
        "Sunny_Count": len(sunny),
        "Cloudy_Count": len(cloudy)
    }


# ==========================================================
# PEAK HOUR ANALYSIS
# ==========================================================
def peak_hour_analysis(df, y_true, y_pred):

    df = df.copy()
    df["Actual"] = y_true
    df["Predicted"] = y_pred

    peak = df[(df["Hour"] >= 10) & (df["Hour"] <= 14)]
    off_peak = df[(df["Hour"] < 10) | (df["Hour"] > 14)]

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    return {
        "Peak_RMSE": rmse(peak["Actual"], peak["Predicted"]) if len(peak) > 0 else None,
        "OffPeak_RMSE": rmse(off_peak["Actual"], off_peak["Predicted"]) if len(off_peak) > 0 else None,
        "Peak_Count": len(peak),
        "OffPeak_Count": len(off_peak)
    }


# ==========================================================
# ERROR DISTRIBUTION
# ==========================================================
def plot_error_distribution(y_true, y_pred):

    errors = np.array(y_true) - np.array(y_pred)

    plt.figure()
    plt.hist(errors, bins=50)
    plt.title("Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()


# ==========================================================
# ERROR TREND
# ==========================================================
def plot_error_trend(y_true, y_pred):

    errors = np.abs(np.array(y_true) - np.array(y_pred))

    plt.figure()
    plt.plot(errors)
    plt.title("Error Trend Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.show()


# ==========================================================
# FULL ANALYSIS PIPELINE (IMPORTANT)
# ==========================================================
def full_analysis(df, y_true, y_pred, model_name="Model"):

    print(f"\n===== {model_name} PERFORMANCE =====")

    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)

    weather = evaluate_weather_performance(df, y_true, y_pred)
    print("Weather Analysis:", weather)

    peak = peak_hour_analysis(df, y_true, y_pred)
    print("Peak Analysis:", peak)

    # Plots
    plot_error_distribution(y_true, y_pred)
    plot_error_trend(y_true, y_pred)

    return {
        "metrics": metrics,
        "weather": weather,
        "peak": peak
    }