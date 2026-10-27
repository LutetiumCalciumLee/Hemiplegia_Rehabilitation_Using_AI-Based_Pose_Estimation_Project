import cv2
from ultralytics import YOLO
import pandas as pd
import os
import json
from datetime import datetime

# =================================================================
# 초기값 설정
# =================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 설정 파일 경로 지정
config_path = os.path.join(script_dir, 'config.json')

# 설정 파일 읽어오기 및 예외 처리
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pose_config = config.get('pose', {})
        # config.json에서 save_cnt 값을 읽어옴 (없으면 기본값 300 적용)
        save_cnt = config.get('save_cnt', 300)
except FileNotFoundError:
    print('config.json 파일이 없어서 기본값으로 실행함')
    pose_config = {'sitting': 'sitting_pose', 'standing': 'standing_pose'}
    save_cnt = 300
except json.JSONDecodeError:
    print('config.json 파일 내용이 비어있거나 형식이 잘못되어 기본값으로 실행함')
    pose_config = {'sitting': 'sitting_pose', 'standing': 'standing_pose'}
    save_cnt = 300

Models_dir = os.path.abspath(os.path.join(script_dir, './Models'))
model_file = os.path.join(Models_dir, 'yolov8n-pose.pt')

save_dir = './pose_img/'
keypoint_path = f'{save_dir}keypoints.csv'

# 메인 저장 폴더 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 설정 파일의 pose 값들을 기반으로 하위 폴더 자동 생성
for pose_name, folder_name in pose_config.items():
    pose_dir = os.path.join(save_dir, folder_name)
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

model = YOLO(model_file)
cap = cv2.VideoCapture(0)

a = 0  
frame_count = 0  
all_data = []

while cap.isOpened():
    flag, frame = cap.read()

    if not flag:
        break

    frame_count += 1

    if frame_count % 3 != 0:
        continue

    results = model(frame, verbose=False)
    
    # 전체 화면에 스켈레톤 그리기
    annotated_frame = results[0].plot(labels=False)

    r = results[0]
    bound_box = r.boxes.xyxy  
    conf = r.boxes.conf.tolist()  
    keypoints = r.keypoints.xyn.tolist()  

    for index, box in enumerate(bound_box):
        if conf[index] > 0.75:
            
            # 밀리초 3자리를 포함하여 파일 이름이 겹치지 않게 방지함
            now = datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            img_name = f'person_{timestamp}.jpg'
            
            output_path = f'{save_dir}{img_name}'
            data = {'image_name': img_name}

            for j in range(len(keypoints[index])):
                data[f'x{j}'] = keypoints[index][j][0]
                data[f'y{j}'] = keypoints[index][j][1]

            all_data.append(data)
            
            # 자르지 않은 깨끗한 전체 화면 원본 저장
            cv2.imwrite(output_path, frame)
            
            a += 1  

            # 설정 파일에서 읽어온 save_cnt 값으로 종료 조건 적용
            if a >= save_cnt:
                break

    # 화면에는 스켈레톤이 덧그려진 전체 화면을 띄움
    display_img = annotated_frame.copy()
    text = f'Saved Count: {a} / {save_cnt}' # 진행 상황을 보기 쉽게 목표 개수도 같이 출력되도록 수정함
    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Saved Person Pose Image', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # 설정 파일에서 읽어온 save_cnt 값으로 종료 조건 적용
    if a >= save_cnt:
        break

print(f'Processed frames: {frame_count}, Saved images: {a}')

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(all_data)

# 기존 CSV 파일이 있는지 확인하여 헤더(열 이름) 추가 여부를 결정함
file_exists = os.path.isfile(keypoint_path)

# mode='a'를 사용하여 파일이 존재하면 맨 아래에 데이터를 이어 붙임
df.to_csv(keypoint_path, mode='a', index=False, header=not file_exists)
print(f'Keypoints saved to {keypoint_path}')