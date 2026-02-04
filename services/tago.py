import requests

BASE_STTN = "http://apis.data.go.kr/1613000/BusSttnInfoInqireService"
BASE_ARVL = "http://apis.data.go.kr/1613000/ArvlInfoInqireService"

def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    return r.json()

def get_nearby_stops(service_key: str, gps_lati: float, gps_long: float, num_rows: int = 10) -> dict:
    # 좌표기반 근접정류소 목록조회: getCrdntPrxmtSttnList :contentReference[oaicite:3]{index=3}
    url = f"{BASE_STTN}/getCrdntPrxmtSttnList"
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": num_rows,
        "_type": "json",
        "gpsLati": gps_lati,
        "gpsLong": gps_long,
    }
    j = _get(url, params)
    items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]

    # 표준화
    out = []
    for it in items:
        out.append({
            "nodeId": it.get("nodeid") or it.get("nodeId"),
            "nodeNm": it.get("nodenm") or it.get("nodeNm"),
            "gpsLati": it.get("gpslati") or it.get("gpsLati"),
            "gpsLong": it.get("gpslong") or it.get("gpsLong"),
        })
    return {"ok": True, "stops": out, "raw": j}

def get_arrivals_by_stop(service_key: str, city_code: str, node_id: str, num_rows: int = 30) -> dict:
    # 정류소별 도착예정정보 목록 조회: getSttnAcctoArvlPrearngeInfoList :contentReference[oaicite:4]{index=4}
    url = f"{BASE_ARVL}/getSttnAcctoArvlPrearngeInfoList"
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": num_rows,
        "_type": "json",
        "cityCode": city_code,
        "nodeId": node_id,
    }
    j = _get(url, params)
    items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]

    out = []
    for it in items:
        # arrtime: 초 단위(문서 샘플 기준) :contentReference[oaicite:5]{index=5}
        arr_sec = it.get("arrtime")
        try:
            arr_min = int(round(int(arr_sec) / 60)) if arr_sec is not None else None
        except Exception:
            arr_min = None

        out.append({
            "routeId": it.get("routeid"),
            "routeNo": it.get("routeno"),
            "routeTp": it.get("routetp"),
            "arrPrevStationCnt": it.get("arrprevstationcnt"),
            "vehicleTp": it.get("vehicletp"),
            "arrTimeSec": arr_sec,
            "arrTimeMin": arr_min,
        })

    # “가장 빨리 오는 버스” ETA(min) 하나 뽑아주기
    eta_min = None
    mins = [x["arrTimeMin"] for x in out if isinstance(x.get("arrTimeMin"), int)]
    if mins:
        eta_min = min(mins)

    return {"ok": True, "arrivals": out, "eta_min": eta_min, "raw": j}