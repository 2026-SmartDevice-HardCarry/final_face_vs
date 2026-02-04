import os
from dotenv import load_dotenv

load_dotenv()

def _f(name: str, default: float):
    v = os.getenv(name)
    return float(v) if v is not None else float(default)

def _i(name: str, default: int):
    v = os.getenv(name)
    return int(v) if v is not None else int(default)

class Config:
    TZ = os.getenv("TZ", "Asia/Seoul")

    OWM_API_KEY = os.getenv("OWM_API_KEY", "")
    HOME_LAT = _f("HOME_LAT", 0.0)
    HOME_LON = _f("HOME_LON", 0.0)

    TAGO_SERVICE_KEY = os.getenv("TAGO_SERVICE_KEY", "")
    TAGO_CITY_CODE = os.getenv("TAGO_CITY_CODE", "")  # optional
    BUS_STOP_LAT = _f("BUS_STOP_LAT", 0.0)
    BUS_STOP_LON = _f("BUS_STOP_LON", 0.0)

    CAM_INDEX = _i("CAM_INDEX", 0)
    CAM_WIDTH = _i("CAM_WIDTH", 640)
    CAM_HEIGHT = _i("CAM_HEIGHT", 360)