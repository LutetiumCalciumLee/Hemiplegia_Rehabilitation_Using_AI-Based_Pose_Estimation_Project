import cv2
from ultralytics import YOLO
import os
import json

# =================================================================
# 초기값 설정
# =================================================================
# 실행 경로 설정 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 설정 파일 읽어오기
config_path = os.path.join(script_dir, 'config.json')

# 에러 방지를 위한 예외 처리 구문 추가
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pose_config = config.get('pose', {})
except FileNotFoundError:
    print('config.json 파일이 없어서 기본값으로 실행함')
    pose_config = {'sitting': 'sitting_pose', 'standing': 'standing_pose'}
except json.JSONDecodeError:
    print('config.json 파일 내용이 비어있거나 형식이 잘못되어 기본값으로 실행함')
    pose_config = {'sitting': 'sitting_pose', 'standing': 'standing_pose'}
    
    

Models_dir = os.path.abspath(os.path.join(script_dir, './Models'))
Videos_dir = os.path.abspath(os.path.join(script_dir, './Sample_Video'))

model_file = os.path.join(Models_dir, 'yolov8n-pose.pt' )
video_file = os.path.join(Videos_dir, 'input', 'driver.mp4' )

# 1. YOLOv8 Pose 모델 불러오기 (미리 학습된 모델 사용)
model = YOLO(model_file) 

# 2. 실시간 카메라 비디오 스트림 열기
cap = cv2.VideoCapture(video_file)  

# 3. 비디오 스트림에서 프레임을 받아와서 포즈 추정
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO Pose 모델로 포즈 추정 수행
    results = model(frame)

    # 추정된 포즈를 그리기
    annotated_frame = results[0].plot()  # 추정된 포즈를 프레임 위에 그림

    # 4. 결과를 화면에 표시
    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)

    # q 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()