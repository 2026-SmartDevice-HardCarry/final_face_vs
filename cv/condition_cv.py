import cv2
import time
import numpy as np
from dataclasses import dataclass

@dataclass
class ConditionState:
    state: str               # tired / tense / neutral / noface / noresponse
    face_detected: bool
    blink_per_min: float
    closed_ratio_10s: float
    head_motion_std: float
    last_update_ts: float

class ConditionEstimatorCV:
    """
    Raspberry Pi 5 친화 버전:
    - Haar cascade 얼굴/눈 검출
    - 10초 윈도우에서:
        * eyes_not_found 비율 -> closed_ratio proxy
        * 얼굴 중심 이동 std -> head motion
        * 눈 검출의 on/off로 blink proxy(간이)
    """
    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 360, fps_hint: int = 15):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps_hint)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.win_sec = 10.0
        self.samples = []  # (t, face_found, eyes_found, face_cx, face_cy)
        self.last_state = "noface"
        self.last_interaction_ts = time.time()

        # baseline (개인 기준선) - 자동 적응
        self.baseline_closed = 0.25
        self.baseline_motion = 6.0

    def mark_interaction(self):
        self.last_interaction_ts = time.time()

    def _append_sample(self, t, face_found, eyes_found, cx, cy):
        self.samples.append((t, face_found, eyes_found, cx, cy))
        # 윈도우 유지
        cut = t - self.win_sec
        while self.samples and self.samples[0][0] < cut:
            self.samples.pop(0)

    def _compute_metrics(self):
        if not self.samples:
            return 0.0, 1.0, 0.0, False

        face_flags = [s[1] for s in self.samples]
        eye_flags  = [s[2] for s in self.samples]
        face_found_ratio = sum(face_flags) / len(face_flags)
        face_detected = face_found_ratio >= 0.3

        if not face_detected:
            return 0.0, 1.0, 0.0, False

        # closed_ratio proxy: 얼굴은 있는데 눈 검출이 안된 비율
        face_indices = [i for i, f in enumerate(face_flags) if f]
        if not face_indices:
            return 0.0, 1.0, 0.0, False

        eyes_missing = 0
        for i in face_indices:
            if not eye_flags[i]:
                eyes_missing += 1
        closed_ratio = eyes_missing / len(face_indices)

        # head motion std: 얼굴 중심점 이동량
        cxs = [self.samples[i][3] for i in face_indices if self.samples[i][3] is not None]
        cys = [self.samples[i][4] for i in face_indices if self.samples[i][4] is not None]
        if len(cxs) >= 3:
            head_motion_std = float(np.sqrt(np.var(cxs) + np.var(cys)))
        else:
            head_motion_std = 0.0

        # blink proxy: eyes_found가 True->False->True 패턴을 blink로 근사
        # 너무 공격적이면 오탐이니 “눈이 사라졌다가 다시 잡힌 횟수”만 센다.
        blinks = 0
        prev = None
        for i in face_indices:
            cur = eye_flags[i]
            if prev is True and cur is False:
                # 닫힘 시작
                pass
            if prev is False and cur is True:
                # 다시 열림 = 1 blink
                blinks += 1
            prev = cur

        # 10초 윈도우 blinks -> per_min
        blink_per_min = (blinks / self.win_sec) * 60.0

        return blink_per_min, float(closed_ratio), float(head_motion_std), True

    def _classify(self, blink_per_min, closed_ratio, head_motion_std, face_detected):
        now = time.time()
        if not face_detected:
            return "noface"

        # NoResponse: 최근 상호작용 없고(예: 12초 이상), 움직임도 낮음
        if (now - self.last_interaction_ts) > 12 and head_motion_std < (self.baseline_motion * 0.7):
            return "noresponse"

        # Tired: 눈감김 프록시가 baseline보다 꽤 높으면
        if closed_ratio > (self.baseline_closed + 0.20):
            return "tired"

        # Tense: 머리 흔들림(불안정) 높으면
        if head_motion_std > (self.baseline_motion + 10.0):
            return "tense"

        return "neutral"

    def _update_baseline(self, closed_ratio, head_motion_std, face_detected):
        # neutral 상태일 때만 천천히 baseline 적응
        if not face_detected:
            return
        alpha = 0.02
        self.baseline_closed = (1 - alpha) * self.baseline_closed + alpha * closed_ratio
        self.baseline_motion = (1 - alpha) * self.baseline_motion + alpha * head_motion_std

    def step(self) -> ConditionState:
        t = time.time()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # 카메라 읽기 실패 시 안전하게 noface
            return ConditionState("noface", False, 0.0, 1.0, 0.0, t)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        face_found = len(faces) > 0
        eyes_found = False
        cx = cy = None

        if face_found:
            # 가장 큰 얼굴 1개만 사용
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx, cy = x + w / 2.0, y + h / 2.0

            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.2, minNeighbors=6, minSize=(20, 20))
            eyes_found = len(eyes) >= 1

        self._append_sample(t, face_found, eyes_found, cx, cy)

        blink_per_min, closed_ratio, head_motion_std, face_detected = self._compute_metrics()
        state = self._classify(blink_per_min, closed_ratio, head_motion_std, face_detected)

        # baseline 업데이트는 neutral일 때만
        if state == "neutral":
            self._update_baseline(closed_ratio, head_motion_std, face_detected)

        self.last_state = state

        return ConditionState(
            state=state,
            face_detected=face_detected,
            blink_per_min=float(round(blink_per_min, 2)),
            closed_ratio_10s=float(round(closed_ratio, 3)),
            head_motion_std=float(round(head_motion_std, 2)),
            last_update_ts=t
        )

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass