from flask import Flask, render_template, jsonify, request, Response, session
from datetime import datetime
import threading
import time
import pytz
import cv2
import numpy as np

from config import Config
from db import init_db, log_event
from cv.condition_cv import ConditionEstimatorCV
# policy 모듈은 ui_mode 결정을 위해 유지하거나, 필요 없다면 제거 가능합니다.
from logic.policy import apply_policy

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.secret_key = "mirror_secret_key_1234"
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
latest_frame = None 

# 라즈베리파이로부터 영상을 받는 입구
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    try:
        # 헤더에서 사용자 ID 추출 및 세션 저장
        user_id = request.headers.get('User-ID', 'Unknown')
        if user_id != "Unknown":
            session['user_id'] = user_id

        img_byte = request.data
        nparr = np.frombuffer(img_byte, np.uint8)
        latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return "OK", 200
    except Exception as e:
        return str(e), 500

# ====== CV 스레드 (피로도 분석) ======
def cv_loop():
    global latest_frame
    est = ConditionEstimatorCV() 
    
    while True:
        # 전역 변수에 저장된 프레임을 분석기로 전달
        st = est.step(external_frame=latest_frame) 
        
        with cv_lock:
            cv_state.update({
                "state": st.state,
                "face_detected": st.face_detected,
                "blink_per_min": st.blink_per_min,
                "closed_ratio_10s": st.closed_ratio_10s,
                "head_motion_std": st.head_motion_std,
                "last_update_ts": st.last_update_ts
            })
        time.sleep(0.1)

threading.Thread(target=cv_loop, daemon=True).start()

# ====== 영상 송출 (브라우저 전송) ======
@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                frame = cv2.flip(latest_frame, 1)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def dashboard():
    current_user = session.get('user_id', 'Unknown')
    now = datetime.now(tz)

    with cv_lock:
        cond = dict(cv_state)

    # UI 모드 결정을 위한 최소한의 정책 적용
    policy = apply_policy(cond["state"])

    # [핵심] 피로도 데이터만 HTML로 전송 (날씨, 버스, 리스크 등 모두 제거)
    return render_template(
        "dashboard.html",
        current_user=current_user,
        now=now.strftime("%Y-%m-%d %H:%M"),
        cond=cond,
        policy=policy
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)