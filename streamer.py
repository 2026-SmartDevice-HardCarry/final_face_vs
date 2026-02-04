import cv2
import requests
import time

# AWS 서버 정보
AWS_IP = "15.164.225.121"
URL = f"http://{AWS_IP}:8080/upload_frame"

# 카메라 설정 (라즈베리파이 5 대응)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

print(f"영상 송신 시작: {URL}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # 이미지 압축 (전송 속도 향상)
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        try:
            # 타임아웃을 2초로 늘려 안정성 확보
            response = requests.post(URL, data=img_encoded.tobytes(), timeout=2.0)
            
            if response.status_code == 200:
                print(".", end="", flush=True) # 전송 성공 시 점 찍기
            else:
                print(f"\n서버 응답 에러: {response.status_code}")
                
        except requests.exceptions.Timeout:
            # 타임아웃 시 멈추지 않고 계속 시도
            print("T", end="", flush=True) 
        except Exception as e:
            print(f"\n전송 중 오류 발생: {e}")
        
        # 전송 부하를 줄이기 위한 미세 대기
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n사용자에 의해 종료되었습니다.")
finally:
    cap.release()
    print("카메라 자원을 해제했습니다.")