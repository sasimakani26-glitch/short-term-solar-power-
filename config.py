import os
import json
import requests
from typing import Dict, Optional
import logging

# Configuration constants
MODEL_PATH = "models/best_model.pkl"
PERFORMANCE_PATH = "models/model_performance.csv"
LATITUDE = 26.85
LONGITUDE = 80.95
TEST_SIZE = 0.2
RANDOM_STATE = 42
LAG_FEATURES = [1, 24]  # lags (in number of timesteps) of Power_Output to include as features
FEATURE_COLUMNS_PATH = "models/feature_columns.json"

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Logger for weather service
logger = logging.getLogger(__name__)

# OpenWeatherMap API endpoint (free tier)
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_live_weather() -> Optional[Dict]:
    """
    Fetch live weather data for the configured location.

    Returns:
        Dictionary with keys: Temperature, Humidity, Cloud_Cover, Wind_Speed.
        Returns None if the API call fails.
    """
    try:
        # Using Open-Meteo API (free, no API key required)
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m",
            "timezone": "auto"
        }

        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        current_weather = data.get("current", {})

        weather_dict = {
            "Temperature": current_weather.get("temperature_2m", 25.0),
            "Humidity": current_weather.get("relative_humidity_2m", 50.0),
            "Cloud_Cover": current_weather.get("cloud_cover", 50.0),
            "Wind_Speed": current_weather.get("wind_speed_10m", 5.0)
        }

        logger.info(f"Weather data fetched successfully: {weather_dict}")
        return weather_dict

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        # Return default values if API fails
        return {
            "Temperature": 25.0,
            "Humidity": 50.0,
            "Cloud_Cover": 50.0,
            "Wind_Speed": 5.0
        }
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing weather data: {e}")
        return {
            "Temperature": 25.0,
            "Humidity": 50.0,
            "Cloud_Cover": 50.0,
            "Wind_Speed": 5.0
        }


