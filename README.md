
<details>
<summary>ENG (English Version)</summary>

# 2D Pose Estimation Dataset for Hemiplegia Rehabilitation (April 2026)

This project focuses on constructing a 2D human pose dataset for future rehabilitation exercise classification for patients with hemiplegia.

In April 2026, I extended the March dataset configuration from two basic postures to five neck movement categories. The existing YOLOv8n-pose-based data collection pipeline was reused, with modifications to `config.json` to support additional movement labels and a larger data collection limit.

## 1. Project Overview

- **Period:** April 2026
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Expand the rehabilitation exercise dataset to include five neck movement categories.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)

## 2. Main Implementation

### Dataset Configuration
- Expanded the posture configuration from 2 to 5 categories.
- Changed the dataset target from basic sitting and standing postures to neck movements.
- Increased the image collection limit from 10 to 200 per execution.
- Reused the existing Python scripts without modifying the pose estimation and data processing logic.

### Pose Estimation & Data Collection
- Used YOLOv8n-pose to detect human poses and extract 17 body keypoints.
- Extracted normalized X and Y coordinates from webcam frames.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates to CSV.

### Dataset Construction
- Configured five posture-specific folders for neck movements.
- Used image filenames and folder locations to assign movement labels.
- Combined normalized keypoint coordinates with corresponding labels.
- Generated a structured dataset for future neck movement classification.

## 3. Neck Movement Categories

Five neck movement categories were defined in `config.json`.

| Label | Description | Folder |
|---|---|---|
| `neck_front` | Forward-facing neck posture | `neck_front_pose` |
| `neck_up` | Upward neck movement | `neck_up_pose` |
| `neck_down` | Downward neck movement | `neck_down_pose` |
| `neck_left` | Leftward neck movement | `neck_left_pose` |
| `neck_right` | Rightward neck movement | `neck_right_pose` |

## 4. Dataset Construction Process

1. **Pose Detection**
   - Detect human poses using the pre-trained YOLOv8n-pose model.
   - Extract 17 body keypoints from video frames.

2. **Keypoint Extraction**
   - Extract normalized X and Y coordinates.
   - Store coordinates as `x0, y0, ..., x16, y16`.

3. **Data Collection**
   - Capture original images from webcam frames.
   - Save image filenames and coordinates to `keypoints.csv`.
   - Apply the configured collection limit of 200 images per execution.

4. **Data Labeling**
   - Organize captured images into five neck movement folders.
   - Assign movement labels based on the corresponding image folders.

5. **Dataset Generation**
   - Combine keypoint coordinates with movement labels.
   - Exclude unlabeled samples.
   - Export the final dataset to `dataset.csv`.

## 5. Dataset Configuration & Structure

### config.json

```json
{
  "save_cnt": 200,
  "pose": {
    "neck_front": "neck_front_pose",
    "neck_up": "neck_up_pose",
    "neck_down": "neck_down_pose",
    "neck_left": "neck_left_pose",
    "neck_right": "neck_right_pose"
  }
}
```

### keypoints.csv
- **Columns:** 35
- **Structure:**
  - `image_name`: Original image filename
  - `x0, y0, ..., x16, y16`: Normalized coordinates of 17 body keypoints

### dataset.csv
- **Columns:** 36
- **Structure:**
  - `image_name`: Original image filename
  - `x0, y0, ..., x16, y16`: Normalized keypoint coordinates
  - `label`: Neck movement classification label

## 6. Project Outcomes

- Expanded the dataset configuration from 2 basic postures to 5 neck movement categories.
- Increased the configurable image collection limit to 200 per execution.
- Reused the existing YOLOv8n-pose data collection and labeling pipeline.
- Configured structured data collection for neck movement classification.
- Prepared the dataset generation process for future rehabilitation exercise recognition model development.

## 7. Generated Images

### Facing Forward
<img width="640" height="480" alt="person_20260422_233310_130" src="https://github.com/user-attachments/assets/e50104df-ea9f-4fcf-8447-0c4063c99862" />

### Looking Up
<img width="640" height="480" alt="person_20260422_233458_795" src="https://github.com/user-attachments/assets/619b32b8-7bdc-4b26-9ced-8ef5b8a36bb9" />

### Looking Down
<img width="640" height="480" alt="person_20260422_233412_643" src="https://github.com/user-attachments/assets/a41f6778-af73-41c4-a031-e38d18b3f0cd" />

### Looking Left
<img width="640" height="480" alt="person_20260422_233543_641" src="https://github.com/user-attachments/assets/67f4bc06-8002-4df8-a3a9-d6ede5300de8" />

### Looking Right
<img width="640" height="480" alt="person_20260422_233653_589" src="https://github.com/user-attachments/assets/a261b3f4-350f-49fd-9270-3a011c0d84e6" />

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 2D Pose Estimation 데이터셋 구축 (2026년 4월)

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

2026년 4월에는 3월에 구축한 데이터 수집 파이프라인을 유지하면서, 기존 앉기와 서 있기 2개 자세를 목 운동 관련 5개 동작으로 확장했습니다. `config.json`을 수정하여 동작별 레이블을 추가하고 이미지 수집 개수를 늘렸습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 4월
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 목 운동 관련 5개 동작을 대상으로 재활 운동 데이터셋 구성 확장
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)

## 2. 주요 구현 내용

### 데이터셋 설정 변경
- 기존 2개 동작 클래스를 5개로 확장
- 앉기와 서 있기 중심의 데이터 수집 설정을 목 운동 동작으로 변경
- 프로그램 1회 실행 시 이미지 수집 목표 개수를 10개에서 200개로 변경
- 기존 Python 코드의 자세 추정 및 데이터 처리 기능을 수정하지 않고 재사용

### 자세 추정 및 데이터 수집
- YOLOv8n-pose를 활용하여 인체 자세 탐지 및 17개 관절 키포인트 추출
- 웹캠 영상에서 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표를 CSV로 저장

### 데이터셋 구축
- 목 운동 동작별 5개 이미지 폴더 설정
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 정규화된 관절 좌표와 동작 레이블 결합
- 향후 목 운동 동작 분류를 위한 구조화된 데이터셋 생성

## 3. 목 운동 동작 분류

`config.json`에 다음 5개 목 운동 동작을 정의했습니다.

| 레이블 | 동작 설명 | 폴더명 |
|---|---|---|
| `neck_front` | 정면 바라보기 | `neck_front_pose` |
| `neck_up` | 고개 위로 향하기 | `neck_up_pose` |
| `neck_down` | 고개 아래로 향하기 | `neck_down_pose` |
| `neck_left` | 고개 왼쪽으로 향하기 | `neck_left_pose` |
| `neck_right` | 고개 오른쪽으로 향하기 | `neck_right_pose` |

## 4. 데이터셋 구축 과정

1. **자세 탐지 (Pose Detection)**
   - 사전 학습된 YOLOv8n-pose 모델을 이용한 인체 자세 탐지
   - 영상 프레임에서 17개 관절 키포인트 추출

2. **키포인트 추출 (Keypoint Extraction)**
   - 정규화된 X, Y 좌표 추출
   - `x0, y0, ..., x16, y16` 형태로 관절 좌표 구성

3. **데이터 수집 (Data Collection)**
   - 웹캠에서 원본 이미지 수집
   - 이미지 파일명과 관절 좌표를 `keypoints.csv`에 저장
   - 설정된 이미지 수집 목표 개수 200개 적용

4. **데이터 레이블링 (Data Labeling)**
   - 수집한 이미지를 목 운동 동작별 5개 폴더로 분류
   - 이미지가 포함된 폴더를 기준으로 동작 레이블 부여

5. **데이터셋 생성 (Dataset Generation)**
   - 관절 좌표와 동작 레이블 결합
   - 레이블이 없는 데이터 제외
   - 최종 학습용 데이터셋을 `dataset.csv`로 저장

## 5. 데이터셋 설정 및 구조

### config.json

```json
{
  "save_cnt": 200,
  "pose": {
    "neck_front": "neck_front_pose",
    "neck_up": "neck_up_pose",
    "neck_down": "neck_down_pose",
    "neck_left": "neck_left_pose",
    "neck_right": "neck_right_pose"
  }
}
```

### keypoints.csv
- **열 개수:** 35개
- **구성:**
  - `image_name`: 원본 이미지 파일명
  - `x0, y0, ..., x16, y16`: 17개 관절의 정규화된 좌표

### dataset.csv
- **열 개수:** 36개
- **구성:**
  - `image_name`: 원본 이미지 파일명
  - `x0, y0, ..., x16, y16`: 정규화된 관절 좌표
  - `label`: 목 운동 동작 분류 레이블

## 6. 프로젝트 성과

- 기존 2개 기본 자세에서 목 운동 관련 5개 동작으로 데이터셋 설정 확장
- 프로그램 1회 실행 시 이미지 수집 목표 개수를 200개로 증가
- 기존 YOLOv8n-pose 기반 데이터 수집 및 레이블링 파이프라인 재사용
- 목 운동 동작 분류를 위한 구조화된 데이터 수집 환경 구성
- 향후 재활 운동 동작 인식 모델 개발을 위한 데이터셋 생성 과정 준비

## 7. 생성 사진
### 정면
<img width="640" height="480" alt="person_20260422_233310_130" src="https://github.com/user-attachments/assets/e50104df-ea9f-4fcf-8447-0c4063c99862" />

### 윗쪽
<img width="640" height="480" alt="person_20260422_233458_795" src="https://github.com/user-attachments/assets/619b32b8-7bdc-4b26-9ced-8ef5b8a36bb9" />

### 아랫쪽
<img width="640" height="480" alt="person_20260422_233412_643" src="https://github.com/user-attachments/assets/a41f6778-af73-41c4-a031-e38d18b3f0cd" />

### 왼쪽
<img width="640" height="480" alt="person_20260422_233543_641" src="https://github.com/user-attachments/assets/67f4bc06-8002-4df8-a3a9-d6ede5300de8" />

### 오른쪽
<img width="640" height="480" alt="person_20260422_233653_589" src="https://github.com/user-attachments/assets/a261b3f4-350f-49fd-9270-3a011c0d84e6" />

</details>
