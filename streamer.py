import cv2
import requests
import time
import face_recognition
import pickle
import os

# --- 1. 얼굴 식별 설정 ---
# 저장된 얼굴 데이터 로드
PKL_PATH = "/home/hardcarry/mirror/registered_faces.pkl"
known_encodings = []
known_names = []

if os.path.exists(PKL_PATH):
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    print(f"얼굴 데이터 로드 완료: {len(known_names)}명")
else:
    print("경고: registered_faces.pkl 파일이 없습니다. 식별 없이 진행합니다.")

# 상태 관리 변수
identified_user = "Unknown"
is_identified = False  # 얼굴 식별 기능 활성화 여부

# --- 2. AWS 서버 설정 ---
AWS_IP = "15.164.225.121" 
BASE_URL = f"http://{AWS_IP}:8080/upload_frame"

# --- 3. 카메라 설정 ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 카메라가 열릴 때까지 잠시 대기 (USB 초기화 시간)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다. USB 포트를 확인하세요.")
    exit()

# [핵심] USB 카메라 전용 MJPG 포맷 및 프레임 속도 설정
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
cap.set(cv2.CAP_PROP_FPS, 30)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"영상 송신 시작: {BASE_URL}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # --- [시나리오 1단계: 얼굴 식별] ---
        # 식별이 되지 않았을 때만 무거운 연산 수행
        if not is_identified and len(known_encodings) > 0:
            # 연산량 감소를 위해 1/4 크기로 축소
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    if True in matches:
                        first_match_index = matches.index(True)
                        identified_user = known_names[first_match_index]
                        is_identified = True # [시나리오 2단계: 식별 기능 끄기]
                        print(f"\n{identified_user}님 안녕하세요! 식별 기능을 종료하고 측정을 시작합니다.")
                        break

        # 이미지 압축 (전송 속도 향상)
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        try:
            # --- [시나리오 3단계: 피로도 측정 데이터 전송] ---
            # 서버에 누가 접속 중인지 헤더나 파라미터로 함께 전송
            headers = {'User-ID': identified_user}
            response = requests.post(BASE_URL, data=img_encoded.tobytes(), headers=headers, timeout=2.0)
            
            if response.status_code == 200:
                if is_identified:
                    print("*", end="", flush=True) 
                else:
                    print(".", end="", flush=True)
                
        except requests.exceptions.Timeout:
            print("T", end="", flush=True) 
        except Exception as e:
            print(f"\n전송 중 오류 발생: {e}")
        
        # CPU 휴식을 위한 대기 (식별 후에는 더 쾌적해짐)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n사용자에 의해 종료되었습니다.")
finally:
    cap.release()
    print("카메라 자원을 해제했습니다.")