import requests

def get_openweather(api_key: str, lat: float, lon: float) -> dict:
    if not api_key:
        return {"ok": False, "error": "OWM_API_KEY missing"}

    # Current Weather API
    current_url = "https://api.openweathermap.org/data/2.5/weather"
    current_params = {
        "lat": lat, "lon": lon,
        "appid": api_key,
        "units": "metric"
    }

    # Forecast API
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    forecast_params = {
        "lat": lat, "lon": lon,
        "appid": api_key,
        "units": "metric"
    }

    try:
        # Get current weather
        current_r = requests.get(current_url, params=current_params, timeout=8)
        if current_r.status_code == 401:
            return {
                "ok": False, 
                "error": "API Key 인증 실패. 키가 유효한지 확인하세요."
            }
        current_r.raise_for_status()
        current_j = current_r.json()

        # Get forecast
        forecast_r = requests.get(forecast_url, params=forecast_params, timeout=8)
        forecast_r.raise_for_status()
        forecast_j = forecast_r.json()

        # Process current data
        cur = current_j
        temp = cur.get("main", {}).get("temp")
        feels_like = cur.get("main", {}).get("feels_like")
        humidity = cur.get("main", {}).get("humidity")
        wind_speed = cur.get("wind", {}).get("speed")

        # Process hourly data (first 12 hours from forecast)
        hourly = forecast_j.get("list", [])[:12]
        pops = [h.get("pop", 0.0) for h in hourly if isinstance(h, dict)]
        precip_prob = max(pops) if pops else 0.0

        return {
            "ok": True,
            "temp": temp,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind": wind_speed,
            "precip_prob": float(precip_prob),
            "raw": {"current": current_j, "forecast": forecast_j}
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return {"ok": False, "error": str(e)}