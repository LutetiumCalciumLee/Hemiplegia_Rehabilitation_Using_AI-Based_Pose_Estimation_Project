import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
import json

# 팝업창을 띄우기 위해 tkinter 기본 창을 생성하되, 불필요한 메인 창은 숨김 처리함
root = tk.Tk()
root.withdraw()

# 프로그램 시작 알림 팝업
messagebox.showinfo('알림', 'Dataset 생성 중')

script_dir = os.path.dirname(os.path.abspath(__file__))

# 설정 파일 읽어오기
config_path = os.path.join(script_dir, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
    pose_config = config.get('pose', {})

# 저장 경로 설정
save_dir = './pose_img/'
keypoint_path = f'{save_dir}keypoints.csv'
dataset_dir = f'{save_dir}/'

# 폴더별 파일 목록을 저장할 딕셔너리 생성
class_files = {}

# 지정한 폴더들을 돌면서 파일 목록을 한 번에 세트(set)로 읽어옴
for label, folder_name in pose_config.items():
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.exists(folder_path):
        class_files[label] = set(os.listdir(folder_path))
    else:
        class_files[label] = set()

# CSV 파일 로드
df = pd.read_csv(keypoint_path)

# 이미지 이름에 따라 레이블을 지정하는 함수 정의
def get_label(image_name):
    # 등록된 라벨과 파일 목록을 확인하여 일치하는 라벨을 반환함
    for label, files in class_files.items():
        if image_name in files:
            return label
    return None  

# CSV 파일에 'label' 열 추가
df['label'] = df['image_name'].apply(get_label)

# 레이블이 없는 데이터 필터링 (None 제거)
df = df.dropna(subset=['label'])

# 새로운 CSV 파일로 저장
df.to_csv(f'{dataset_dir}dataset.csv', index=False)

# 프로그램 종료 알림 팝업
messagebox.showinfo('알림', 'Dataset 생성 종료')