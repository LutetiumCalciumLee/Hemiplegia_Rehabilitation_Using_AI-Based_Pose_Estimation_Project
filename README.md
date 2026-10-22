
<details>
<summary>ENG (English Version)</summary>

# 2D Pose Estimation Dataset for Hemiplegia Rehabilitation

This project focuses on constructing a 2D human pose dataset for future rehabilitation exercise classification for patients with hemiplegia.

In March 2026, I developed a data collection pipeline using a pre-trained YOLOv8n-pose model to extract body keypoints, save normalized coordinates, and generate labeled CSV datasets.

## 1. Project Overview

- **Period:** March 2026
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Build a structured dataset for future rehabilitation exercise classification.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)

## 2. Main Implementation

### Pose Estimation
- Implemented 2D human pose estimation using YOLOv8n-pose.
- Extracted 17 body keypoints from video frames.
- Visualized detected keypoints and skeleton connections using OpenCV.
- Tested pose estimation using a video file.

### Data Collection
- Implemented webcam-based pose data collection.
- Extracted normalized X and Y coordinates for 17 body keypoints.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates.
- Configured posture classes and image collection limits using `config.json`.

### Dataset Construction
- Organized collected images into posture-specific folders.
- Assigned posture labels based on image filenames and folder locations.
- Combined keypoint coordinates with corresponding posture labels.
- Excluded unlabeled data and generated the final CSV dataset.

## 3. Dataset Construction Process

1. **Pose Detection**
   - Detect human poses using YOLOv8n-pose.
   - Extract 17 body keypoints from video frames.

2. **Keypoint Extraction**
   - Extract normalized X and Y coordinates.
   - Store keypoints as `x0, y0, ..., x16, y16`.

3. **Data Collection**
   - Capture original images from webcam frames.
   - Save image filenames and keypoint coordinates to `keypoints.csv`.

4. **Data Labeling**
   - Organize images into `sitting_pose` and `standing_pose` folders.
   - Assign `sitting` and `standing` labels by matching image filenames.

5. **Dataset Generation**
   - Combine keypoint coordinates with posture labels.
   - Remove unlabeled samples.
   - Export the final labeled dataset to `dataset.csv`.

## 4. Dataset Results

**Collection Date:** March 22, 2026

### keypoints.csv
- **Total Samples:** 100
- **Columns:** 35
- **Structure:**
  - `image_name`: Original image filename
  - `x0, y0, ..., x16, y16`: Normalized coordinates of 17 body keypoints

### dataset.csv
- **Total Samples:** 100
- **Columns:** 36
- **Structure:**
  - `image_name`: Original image filename
  - `x0, y0, ..., x16, y16`: Normalized keypoint coordinates
  - `label`: Posture classification label

### Class Distribution
- **Standing:** 50 samples
- **Sitting:** 50 samples

## 5. Project Files

- **1.2D_Pose_Estimation_test.py**
  - Tests pose estimation using a video file.
  - Visualizes detected body keypoints and skeleton connections.

- **2.2D_Pose_Save.py**
  - Captures webcam frames and extracts 17 body keypoints.
  - Saves original images and normalized coordinates to `keypoints.csv`.

- **3.2D_Pose_Dataset_Save.py**
  - Assigns posture labels based on classified image folders.
  - Generates the final labeled `dataset.csv`.

- **config.json**
  - Defines posture classes and image collection limits.

- **Models/yolov8n-pose.pt**
  - Pre-trained YOLOv8n pose estimation model.

- **keypoints.csv**
  - Contains image filenames and normalized body keypoint coordinates.

- **dataset.csv**
  - Contains keypoint coordinates and corresponding posture labels.

## 6. Project Outcomes

- Implemented a 2D pose estimation and data collection pipeline using YOLOv8n-pose.
- Automated extraction and CSV storage of 17 body keypoints.
- Constructed a labeled dataset containing 100 samples across two posture classes.
- Prepared structured pose data for future rehabilitation exercise classification model development.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 2D Pose Estimation 데이터셋 구축

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

2026년 3월에는 사전 학습된 YOLOv8n-pose 모델을 활용하여 인체 관절 좌표를 추출하고, 정규화된 좌표를 저장한 뒤 동작별 레이블이 포함된 CSV 데이터셋을 생성하는 파이프라인을 구현했습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 3월
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 향후 재활 운동 동작 분류를 위한 구조화된 학습용 데이터셋 구축
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)

## 2. 주요 구현 내용

### 자세 추정 (Pose Estimation)
- YOLOv8n-pose를 활용한 2D 인체 자세 추정 구현
- 영상 프레임에서 17개 인체 관절 키포인트 추출
- OpenCV를 활용한 관절 좌표 및 스켈레톤 시각화
- 동영상 파일을 이용한 자세 추정 기능 테스트

### 데이터 수집 (Data Collection)
- 웹캠 기반 자세 데이터 수집 기능 구현
- 17개 관절 키포인트의 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표 저장
- `config.json`을 이용한 동작 클래스 및 이미지 수집 개수 설정

### 데이터셋 구축 (Dataset Construction)
- 수집한 이미지를 자세별 폴더로 분류
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 관절 좌표와 해당 동작 레이블 결합
- 레이블이 없는 데이터를 제외하고 최종 CSV 데이터셋 생성

## 3. 데이터셋 구축 과정

1. **자세 탐지 (Pose Detection)**
   - YOLOv8n-pose 모델을 이용한 인체 자세 탐지
   - 영상 프레임에서 17개 관절 키포인트 추출

2. **키포인트 추출 (Keypoint Extraction)**
   - 정규화된 X, Y 좌표 추출
   - `x0, y0, ..., x16, y16` 형태로 관절 좌표 구성

3. **데이터 수집 (Data Collection)**
   - 웹캠에서 원본 이미지 수집
   - 이미지 파일명과 관절 좌표를 `keypoints.csv`에 저장

4. **데이터 레이블링 (Data Labeling)**
   - 이미지를 `sitting_pose`, `standing_pose` 폴더로 분류
   - 이미지 파일명을 기준으로 `sitting`, `standing` 레이블 부여

5. **데이터셋 생성 (Dataset Generation)**
   - 관절 좌표와 동작 레이블 결합
   - 레이블이 없는 데이터 제외
   - 최종 학습용 데이터셋을 `dataset.csv`로 저장

## 4. 데이터셋 구축 결과

**데이터 수집일:** 2026년 3월 22일

### keypoints.csv
- **전체 데이터:** 100개
- **열 개수:** 35개
- **구성:**
  - `image_name`: 원본 이미지 파일명
  - `x0, y0, ..., x16, y16`: 17개 관절의 정규화된 좌표

### dataset.csv
- **전체 데이터:** 100개
- **열 개수:** 36개
- **구성:**
  - `image_name`: 원본 이미지 파일명
  - `x0, y0, ..., x16, y16`: 정규화된 관절 좌표
  - `label`: 자세 분류 레이블

### 동작별 데이터 분포
- **서 있기 (Standing):** 50개
- **앉기 (Sitting):** 50개

## 5. 프로젝트 파일 구성

- **1.2D_Pose_Estimation_test.py**
  - 동영상 파일을 이용한 자세 추정 기능 테스트
  - 관절 좌표 및 스켈레톤 시각화

- **2.2D_Pose_Save.py**
  - 웹캠 영상에서 17개 관절 키포인트 추출
  - 원본 이미지와 정규화된 관절 좌표를 `keypoints.csv`에 저장

- **3.2D_Pose_Dataset_Save.py**
  - 분류된 이미지 폴더를 기준으로 동작 레이블 부여
  - 최종 학습용 `dataset.csv` 생성

- **config.json**
  - 동작 클래스 및 이미지 수집 개수 설정

- **Models/yolov8n-pose.pt**
  - 사전 학습된 YOLOv8n 자세 추정 모델

- **keypoints.csv**
  - 이미지 파일명과 정규화된 관절 좌표 저장

- **dataset.csv**
  - 관절 좌표와 동작별 레이블이 포함된 데이터셋

## 6. 프로젝트 성과

- YOLOv8n-pose를 활용한 2D 자세 추정 및 데이터 수집 파이프라인 구현
- 17개 인체 관절 키포인트 추출 및 CSV 저장 자동화
- 앉기와 서 있기 2개 동작에 대해 총 100개 데이터셋 구축
- 향후 재활 운동 동작 분류 모델 개발을 위한 구조화된 학습 데이터 준비

</details>
