<details>
<summary>ENG (English Version)</summary>

# 2D Pose Estimation Dataset for Hemiplegia Rehabilitation (July 2026)

This project focuses on constructing a 2D human pose dataset for future rehabilitation exercise classification for patients with hemiplegia.

In July 2026, I changed the dataset configuration from two arm movement categories to two left and right oblique posture categories. The existing YOLOv8n-pose-based data collection pipeline was reused, with modifications to `config.json` to support the new posture labels while maintaining the existing image collection limit.

## 1. Project Overview

- **Period:** July 2026
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Configure the rehabilitation exercise dataset for two left and right oblique posture categories.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)

## 2. Main Implementation

### Dataset Configuration
- Changed the dataset configuration from 2 arm movement categories to 2 oblique posture categories.
- Updated `config.json` to define left and right oblique posture labels and corresponding image folders.
- Maintained the image collection limit of 200 per execution.
- Used the existing pose estimation and dataset generation pipeline with the updated configuration.

### Pose Estimation & Data Collection
- Used YOLOv8n-pose to detect human poses and extract 17 body keypoints.
- Extracted normalized X and Y coordinates from webcam frames.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates.
- Stored extracted keypoint coordinates in CSV format.

### Dataset Construction
- Configured two posture-specific folders for left and right oblique postures.
- Used image filenames and folder locations to assign posture labels.
- Combined normalized keypoint coordinates with corresponding labels.
- Prepared structured datasets for future posture classification.

## 3. Oblique Posture Categories

Two oblique posture categories were defined in `config.json`.

| Label | Description | Folder |
|---|---|---|
| `obl_right` | Right Oblique Posture | `obl_right_pose` |
| `obl_left` | Left Oblique Posture | `obl_left_pose` |

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
   - Organize captured images into two oblique posture folders.
   - Assign posture labels based on the corresponding image folders.

5. **Dataset Generation**
   - Combine keypoint coordinates with posture labels.
   - Exclude unlabeled samples.
   - Export the final dataset to `dataset.csv`.

## 5. Dataset Configuration & Structure

### config.json

```json
{
  "save_cnt": 200,
  "pose": {
    "obl_right": "obl_right_pose",
    "obl_left": "obl_left_pose"
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
  - `label`: Oblique posture classification label

## 6. Project Outcomes

- Changed the dataset configuration from 2 arm movement categories to 2 oblique posture categories.
- Maintained the configurable image collection limit of 200 per execution.
- Adapted the existing YOLOv8n-pose data collection and labeling pipeline for left and right oblique postures.
- Configured structured data collection for oblique posture classification.
- Prepared the dataset generation process for future rehabilitation exercise recognition model development.

## 7. Generated Images

### Right Oblique Posture

<img width="640" height="480" alt="person_20260720_213812_717" src="https://github.com/user-attachments/assets/06a30c6f-133a-47fa-b262-144fb694fbe6" />

### Left Oblique Posture

<img width="640" height="480" alt="person_20260720_213854_249" src="https://github.com/user-attachments/assets/8225f25c-e022-4e50-b948-2cd9320aeaeb" />

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 2D Pose Estimation 데이터셋 구축 (2026년 7월)

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

2026년 7월에는 기존 데이터 수집 파이프라인을 활용하여 팔 운동 중심의 데이터셋 설정을 좌우 기울임 자세 2개 동작으로 변경했습니다. `config.json`을 수정하여 좌우 기울임 자세 레이블과 해당 이미지 폴더를 설정하고, 기존 이미지 수집 목표 개수를 유지했습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 7월
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 좌우 기울임 자세 2개 동작을 대상으로 재활 운동 데이터셋 구성 변경
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)

## 2. 주요 구현 내용

### 데이터셋 설정 변경
- 기존 팔 운동 2개 동작 클래스를 좌우 기울임 자세 2개 동작으로 변경
- `config.json`을 수정하여 좌우 기울임 자세 레이블 및 이미지 폴더 설정
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 자세 추정 및 데이터셋 생성 파이프라인에 변경된 설정 적용

### 자세 추정 및 데이터 수집
- YOLOv8n-pose를 활용하여 인체 자세 탐지 및 17개 관절 키포인트 추출
- 웹캠 영상에서 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표 저장
- 추출한 관절 좌표를 CSV 형식으로 저장

### 데이터셋 구축
- 좌우 기울임 자세별 2개 이미지 폴더 설정
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 정규화된 관절 좌표와 동작 레이블 결합
- 향후 좌우 기울임 자세 분류를 위한 구조화된 데이터셋 구성

## 3. 좌우 기울임 자세 분류

`config.json`에 다음 2개 자세를 정의했습니다.

| 레이블 | 동작 설명 | 폴더명 |
|---|---|---|
| `obl_right` | 우측 기울임 자세 | `obl_right_pose` |
| `obl_left` | 좌측 기울임 자세 | `obl_left_pose` |

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
   - 수집한 이미지를 좌우 기울임 자세별 2개 폴더로 분류
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
    "obl_right": "obl_right_pose",
    "obl_left": "obl_left_pose"
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
  - `label`: 좌우 기울임 자세 분류 레이블

## 6. 프로젝트 성과

- 기존 팔 운동 2개 동작에서 좌우 기울임 자세 2개 동작으로 데이터셋 설정 변경
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 YOLOv8n-pose 기반 데이터 수집 및 레이블링 파이프라인을 좌우 기울임 자세에 적용
- 좌우 기울임 자세 분류를 위한 구조화된 데이터 수집 환경 구성
- 향후 재활 운동 동작 인식 모델 개발을 위한 데이터셋 생성 과정 준비

## 7. 생성 사진

### 우측 기울임 자세

<img width="640" height="480" alt="person_20260720_213812_717" src="https://github.com/user-attachments/assets/06a30c6f-133a-47fa-b262-144fb694fbe6" />

### 좌측 기울임 자세

<img width="640" height="480" alt="person_20260720_213854_249" src="https://github.com/user-attachments/assets/8225f25c-e022-4e50-b948-2cd9320aeaeb" />

</details>
