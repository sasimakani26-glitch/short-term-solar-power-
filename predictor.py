from config import fetch_live_weather
import joblib
import config

def predict_power():

    model = joblib.load(config.MODEL_PATH)
    weather = fetch_live_weather()

    feature_order = [
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Wind_Speed",

    ]

    input_data = [[weather[f] for f in feature_order]]
    prediction = model.predict(input_data)[0]

    return weather, prediction