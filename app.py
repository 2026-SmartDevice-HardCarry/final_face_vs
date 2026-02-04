from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime
import json
import threading
import time
import pytz
import cv2

from config import Config
from db import init_db, get_stat, set_stat, log_event
from services.openweather import get_openweather
from services.tago import get_nearby_stops, get_arrivals_by_stop
from cv.condition_cv import ConditionEstimatorCV

from logic.ai_commute import success_prob
from logic.ai_behavior import laplace_prob, risk_level
from logic.ai_checklist import order_checklist
from logic.policy import apply_policy
from logic.briefing import make_briefing

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
init_db()

tz = pytz.timezone(Config.TZ)

# ====== 글로벌 공유 자원 ======
cv_lock = threading.Lock()
cv_state = {
    "state": "noface",
    "face_detected": False,
    "blink_per_min": 0.0,
    "closed_ratio_10s": 1.0,
    "head_motion_std": 0.0,
    "last_update_ts": 0.0
}

# 실시간 영상을 공유하기 위한 전역 변수
latest_frame = None 

def iso_now():
    return datetime.now(tz).isoformat(timespec="seconds")

def safe_int(s: str, default=0):
    try: return int(s)
    except: return default

def parse_hhmm(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh)*60 + int(mm)

# ====== CV 스레드 (카메라 독점 사용 및 프레임 공유) ======
def cv_loop():
    global latest_frame
    # 여기서만 카메라를 딱 한 번 엽니다.
    est = ConditionEstimatorCV(
        cam_index=Config.CAM_INDEX,
        width=Config.CAM_WIDTH,
        height=Config.CAM_HEIGHT,
        fps_hint=15
    )
    
    last_logged = None
    
    while True:
        # 1. AI 분석 수행
        st = est.step()
        
        # 2. 분석된 영상 프레임 가져오기 (이 기능이 ConditionEstimatorCV에 있어야 함)
        # 만약 est.step()이 프레임을 반환하지 않는다면 아래와 같이 est 내부 cap에 접근 시도
        if hasattr(est, 'cap'):
            success, frame = est.cap.read()
            if success:
                latest_frame = frame # 최신 프레임을 전역 변수에 저장
        
        with cv_lock:
            cv_state.update({
                "state": st.state,
                "face_detected": st.face_detected,
                "blink_per_min": st.blink_per_min,
                "closed_ratio_10s": st.closed_ratio_10s,
                "head_motion_std": st.head_motion_std,
                "last_update_ts": st.last_update_ts
            })

        if last_logged != st.state:
            log_event(iso_now(), "condition_detected", json.dumps({
                "condition_state": st.state,
                "blink_per_min": st.blink_per_min,
                "closed_ratio_10s": st.closed_ratio_10s,
                "head_motion_std": st.head_motion_std
            }, ensure_ascii=False))
            last_logged = st.state

        time.sleep(0.05)

threading.Thread(target=cv_loop, daemon=True).start()

# ====== 영상 송출 (공유된 프레임을 브라우저로 전송) ======
@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                # 거울 반전
                frame = cv2.flip(latest_frame, 1)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1) # 전송 부하 감소
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ... (기존 api_interaction, api_nearby, api_arrivals 코드와 동일하므로 중략) ...

@app.route("/")
def dashboard():
    now = datetime.now(tz)
    now_min = now.hour * 60 + now.minute

    # 1. CV 상태 가져오기
    with cv_lock:
        cond = dict(cv_state)

    # 2. 정책 적용 (UI 모드 결정)
    policy = apply_policy(cond["state"])

    # ---- 3. 날씨 정보 (터미널 로그 출력 기능 추가) ----
    weather = get_openweather(Config.OWM_API_KEY, Config.HOME_LAT, Config.HOME_LON)
    
    # 터미널 출력용 로그
    print("\n" + "☀️" + "-"*30)
    if weather.get("ok"):
        temp = weather.get("temp")
        precip = float(weather.get("precip_prob", 0.0))
        print(f" [날씨 수신 성공] 현재 온도: {temp}°C | 강수확률: {int(precip*100)}%")
        precip_prob = precip
    else:
        print(f" [날씨 수신 실패] 에러: {weather.get('error')}")
        precip_prob = 0.0
    print("-" * 32 + "\n")

    rain_like = precip_prob >= 0.5

    # 4. 버스 정보 (TAGO)
    eta_min = None
    chosen_stop = None
    arrivals_preview = []
    city_code = Config.TAGO_CITY_CODE or ""
    try:
        near = get_nearby_stops(Config.TAGO_SERVICE_KEY, Config.BUS_STOP_LAT, Config.BUS_STOP_LON, num_rows=8)
        if near.get("ok") and near["stops"]:
            chosen_stop = near["stops"][0]
            if city_code:
                arr = get_arrivals_by_stop(Config.TAGO_SERVICE_KEY, city_code, chosen_stop["nodeId"], num_rows=20)
                eta_min = arr.get("eta_min")
                arrivals_preview = (arr.get("arrivals") or [])[:5]
    except Exception: pass

    # 5. 교통 및 성공 확률 계산
    avg_depart = get_stat("avg_departure_hhmm", "08:10")
    late_7 = safe_int(get_stat("late_count_7days", "0"), 0)
    depart_delay = max(now_min - parse_hhmm(avg_depart), 0)
    congestion = 0.5 if eta_min is None else min(max((eta_min - 5) / 15.0, 0.0), 1.0)
    
    risk_now = success_prob(congestion, late_7, depart_delay)
    risk_early = success_prob(congestion, late_7, max(depart_delay - 5, 0))

    # 6. 체크리스트 및 브리핑
    base_items = ["차키", "지갑", "휴대폰", "우산"]
    miss_freq = {k: safe_int(get_stat(f"miss_{k}", "0")) for k in ["car_key", "wallet", "phone", "umbrella"]}
    # (주의: db key 이름이 miss_car_key 형태이므로 딕셔너리 키 매칭 확인 필요)
    
    checklist_order = order_checklist(base_items, {}, {"rain": rain_like}) # 간단화 버전
    brief = make_briefing({
        "success_prob_now": risk_now["p"],
        "success_prob_early": risk_early["p"],
        "recommend_depart_in_min": 5,
        "precip_prob": precip_prob,
        "eta_min": eta_min,
        "checklist_risks": []
    })

    # 7. HTML로 데이터 전송
    return render_template(
        "dashboard.html",
        now=now.strftime("%Y-%m-%d %H:%M"),
        cond=cond,
        policy=policy,
        weather=weather if weather.get("ok") else {"temp": None, "feels_like": None, "precip_prob": 0.0},
        precip_prob=precip_prob,
        stop=chosen_stop,
        city_code=city_code,
        eta_min=eta_min,
        arrivals_preview=arrivals_preview,
        risk_now=risk_now,
        risk_early=risk_early,
        checklist_order=checklist_order,
        checklist_risks=[],
        briefing=brief
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)