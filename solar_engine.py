def solar_system_engine(predicted_power, actual_power, previous_power, weather):

    weather = weather or {}
    alerts = []

    # ---------------- ERROR & CONFIDENCE ----------------
    error = abs(actual_power - predicted_power)
    error_percentage = (error / max(actual_power, 1)) * 100

    # confidence decreases when error increases
    confidence = max(0, 100 - error_percentage)

    # ---------------- ALERTS ----------------
    if previous_power and (previous_power - predicted_power) / max(previous_power, 1) > 0.3:
        alerts.append("Sudden Power Drop")

    if predicted_power < 50:
        alerts.append("Low Power")

    if error_percentage > 30:
        alerts.append("High Prediction Error")

    if confidence < 60:
        alerts.append("Low Prediction Confidence")

    if weather.get("Cloud_Cover", 0) > 80:
        alerts.append("Heavy Clouds")

    if weather.get("Wind_Speed", 0) > 15:
        alerts.append("High Wind")

    if weather.get("Temperature", 25) > 40:
        alerts.append("High Temperature")

    # ---------------- ENERGY ----------------
    ghi = weather.get("GHI", 500)
    temp = weather.get("Temperature", 25)
    cloud = weather.get("Cloud_Cover", 0)

    sunlight = ghi * 10
    efficiency = 0.18 * (1 - max(0, temp - 25) * 0.004)
    panel_output = sunlight * efficiency

    # ---------------- CLOUD IMPACT ----------------
    cloudy_factor = 1 - (cloud / 100) * 0.75
    expected_clear_output = panel_output / max(cloudy_factor, 0.1)

    cloud_loss = expected_clear_output - panel_output
    cloud_impact_percentage = (cloud_loss / max(expected_clear_output, 1)) * 100

    if cloud_impact_percentage > 40:
        alerts.append("Severe Cloud Impact")
    elif cloud_impact_percentage > 20:
        alerts.append("Moderate Cloud Impact")

    if not alerts:
        alerts.append("Normal")

    # ---------------- OUTPUT ----------------
    return {
        "alerts": alerts,
        "metrics": {
            "Predicted": round(predicted_power, 2),
            "Actual": round(actual_power, 2),
            "Error": round(error, 2),
            "Error_%": round(error_percentage, 2),
            "Confidence_%": round(confidence, 2)
        },
        "energy": {
            "Panel_Output_W": round(panel_output, 2),
            "Cloud_Impact_%": round(cloud_impact_percentage, 2),
            "Sunlight_Wm2": round(sunlight, 2)
        }
    }