
<details>
<summary>ENG (English Version)</summary>

# 2D Pose Estimation Dataset for Hemiplegia Rehabilitation (June 2026)

This project focuses on constructing a 2D human pose dataset for future rehabilitation exercise classification for patients with hemiplegia.

In June 2026, I changed the dataset configuration from four shoulder movement categories to two arm movement categories. The existing YOLOv8n-pose-based data collection pipeline was reused, with modifications to `config.json` to support arm movement labels while maintaining the existing image collection limit.

## 1. Project Overview

- **Period:** June 2026
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Configure the rehabilitation exercise dataset for two arm movement categories.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)

## 2. Main Implementation

### Dataset Configuration
- Changed the dataset configuration from 4 shoulder movement categories to 2 arm movement categories.
- Updated `config.json` to define arm movement labels and corresponding image folders.
- Maintained the image collection limit of 200 per execution.
- Used the existing pose estimation and dataset generation pipeline with the updated configuration.

### Pose Estimation & Data Collection
- Used YOLOv8n-pose to detect human poses and extract 17 body keypoints.
- Extracted normalized X and Y coordinates from webcam frames.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates to CSV.

### Dataset Construction
- Configured two posture-specific folders for arm movements.
- Used image filenames and folder locations to assign movement labels.
- Combined normalized keypoint coordinates with corresponding labels.
- Prepared structured datasets for future arm movement classification.

## 3. Arm Movement Categories

Two arm movement categories were defined in `config.json`.

| Label | Description | Folder |
|---|---|---|
| `arm_up` | Raised Arm Position | `arm_up_pose` |
| `arm_down` | Lowered Arm Position | `arm_down_pose` |

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
   - Organize captured images into two arm movement folders.
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
    "arm_up": "arm_up_pose",
    "arm_down": "arm_down_pose"
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
  - `label`: Arm movement classification label

## 6. Project Outcomes

- Changed the dataset configuration from 4 shoulder movement categories to 2 arm movement categories.
- Maintained the configurable image collection limit of 200 per execution.
- Adapted the existing YOLOv8n-pose data collection and labeling pipeline for arm movements.
- Configured structured data collection for arm movement classification.
- Prepared the dataset generation process for future rehabilitation exercise recognition model development.

## 7. Generated Images

### Raised Arm Position

<img width="640" height="480" alt="person_20260619_225241_629" src="https://github.com/user-attachments/assets/29be91a6-7a74-438a-82c5-3e216cbf7ddc" />

### Lowered Arm Position

<img width="640" height="480" alt="person_20260619_225116_023" src="https://github.com/user-attachments/assets/5af85853-7192-4422-b819-8da748bb485a" />

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 2D Pose Estimation 데이터셋 구축 (2026년 6월)

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

2026년 6월에는 기존 데이터 수집 파이프라인을 활용하여 어깨 운동 중심의 데이터셋 설정을 팔 운동 관련 2개 동작으로 변경했습니다. `config.json`을 수정하여 팔 운동 레이블과 해당 이미지 폴더를 설정하고, 기존 이미지 수집 목표 개수를 유지했습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 6월
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 팔 운동 관련 2개 동작을 대상으로 재활 운동 데이터셋 구성 변경
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)

## 2. 주요 구현 내용

### 데이터셋 설정 변경
- 기존 어깨 운동 4개 동작 클래스를 팔 운동 2개 동작으로 변경
- `config.json`을 수정하여 팔 운동 레이블 및 이미지 폴더 설정
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 자세 추정 및 데이터셋 생성 파이프라인에 변경된 설정 적용

### 자세 추정 및 데이터 수집
- YOLOv8n-pose를 활용하여 인체 자세 탐지 및 17개 관절 키포인트 추출
- 웹캠 영상에서 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표를 CSV로 저장

### 데이터셋 구축
- 팔 운동 동작별 2개 이미지 폴더 설정
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 정규화된 관절 좌표와 동작 레이블 결합
- 향후 팔 운동 동작 분류를 위한 구조화된 데이터셋 구성

## 3. 팔 운동 동작 분류

`config.json`에 다음 2개 팔 운동 동작을 정의했습니다.

| 레이블 | 동작 설명 | 폴더명 |
|---|---|---|
| `arm_up` | 팔 올리기 | `arm_up_pose` |
| `arm_down` | 팔 내리기 | `arm_down_pose` |

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
   - 수집한 이미지를 팔 운동 동작별 2개 폴더로 분류
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
    "arm_up": "arm_up_pose",
    "arm_down": "arm_down_pose"
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
  - `label`: 팔 운동 동작 분류 레이블

## 6. 프로젝트 성과

- 기존 어깨 운동 4개 동작에서 팔 운동 관련 2개 동작으로 데이터셋 설정 변경
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 YOLOv8n-pose 기반 데이터 수집 및 레이블링 파이프라인을 팔 운동에 적용
- 팔 운동 동작 분류를 위한 구조화된 데이터 수집 환경 구성
- 향후 재활 운동 동작 인식 모델 개발을 위한 데이터셋 생성 과정 준비

## 7. 생성 사진

### 팔 올리기

<img width="640" height="480" alt="person_20260619_225241_629" src="https://github.com/user-attachments/assets/29be91a6-7a74-438a-82c5-3e216cbf7ddc" />

### 팔 내리기

<img width="640" height="480" alt="person_20260619_225116_023" src="https://github.com/user-attachments/assets/5af85853-7192-4422-b819-8da748bb485a" />

</details>
