import requests

def get_openweather(api_key: str, lat: float, lon: float) -> dict:
    if not api_key:
        return {"ok": False, "error": "OWM_API_KEY missing"}

    # One Call 3.0 API 주소
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon,
        "appid": api_key,
        "units": "metric",
        "exclude": "minutely,daily,alerts"
    }

    try:
        r = requests.get(url, params=params, timeout=8)
        
        # 401 에러(인증 실패) 발생 시 서버가 죽지 않게 처리
        if r.status_code == 401:
            return {
                "ok": False, 
                "error": "API Key 인증 실패. 3.0 결제 등록 확인 또는 키 활성화 대기 필요."
            }
            
        r.raise_for_status()
        j = r.json()

        cur = j.get("current", {})
        hourly = (j.get("hourly") or [])[:12]

        # 다음 2시간 강수확률(pop) 중 최대값 사용
        pops = [h.get("pop", 0.0) for h in hourly[:2] if isinstance(h, dict)]
        precip_prob = max(pops) if pops else 0.0

        return {
            "ok": True,
            "temp": cur.get("temp"),
            "feels_like": cur.get("feels_like"),
            "humidity": cur.get("humidity"),
            "wind": cur.get("wind_speed"),
            "precip_prob": float(precip_prob),
            "raw": j
        }
    except Exception as e:
        # 어떤 에러가 발생해도 딕셔너리를 반환하여 app.py의 붕괴 방지
        print(f"Weather API Error: {e}")
        return {"ok": False, "error": str(e)}