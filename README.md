
<details>
<summary>ENG (English Version)</summary>

# 2D Pose Estimation Dataset for Hemiplegia Rehabilitation (May 2026)

This project focuses on constructing a 2D human pose dataset for future rehabilitation exercise classification for patients with hemiplegia.

In May 2026, I extended the existing dataset collection configuration to four shoulder movement categories. The YOLOv8n-pose-based data collection pipeline was used with modifications to `config.json` to support shoulder movement labels while maintaining the existing image collection limit.

## 1. Project Overview

- **Period:** May 2026
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Expand the rehabilitation exercise dataset to include four shoulder movement categories.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)

## 2. Main Implementation

### Dataset Configuration
- Changed the dataset configuration from 5 neck movement categories to 4 shoulder movement categories.
- Updated `config.json` to define shoulder movement labels and corresponding image folders.
- Maintained the image collection limit of 200 per execution.
- Used the existing pose estimation and dataset generation pipeline with the updated configuration.

### Pose Estimation & Data Collection
- Used YOLOv8n-pose to detect human poses and extract 17 body keypoints.
- Extracted normalized X and Y coordinates from webcam frames.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates to CSV.

### Dataset Construction
- Configured four posture-specific folders for shoulder movements.
- Used image filenames and folder locations to assign movement labels.
- Combined normalized keypoint coordinates with corresponding labels.
- Prepared structured datasets for future shoulder movement classification.

## 3. Shoulder Movement Categories

Four shoulder movement categories were defined in `config.json`.

| Label | Description | Folder |
|---|---|---|
| `should_front` | Front Position | `should_front_pose` |
| `should_up` | Upward Position | `should_up_pose` |
| `should_forward` | Forward Position | `should_forward_pose` |
| `should_down` | Downward Position | `should_down_pose` |

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
   - Organize captured images into four shoulder movement folders.
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
    "should_front": "should_front_pose",
    "should_up": "should_up_pose",
    "should_forward": "should_forward_pose",
    "should_down": "should_down_pose"
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
  - `label`: Shoulder movement classification label

## 6. Project Outcomes

- Changed the dataset configuration from 5 neck movement categories to 4 shoulder movement categories.
- Maintained the configurable image collection limit of 200 per execution.
- Adapted the existing YOLOv8n-pose data collection and labeling pipeline for shoulder movements.
- Configured structured data collection for shoulder movement classification.
- Prepared the dataset generation process for future rehabilitation exercise recognition model development.

## 7. Generated Images

### Front Position

<img width="640" height="480" alt="person_20260521_223259_957" src="https://github.com/user-attachments/assets/7e7148de-cba2-43a2-b04f-566b47899d54" />

### Upward Position

<img width="640" height="480" alt="person_20260521_223435_858" src="https://github.com/user-attachments/assets/ed495dc4-4ac3-4333-be45-56549064e6da" />

### Forward Position

<img width="640" height="480" alt="person_20260521_223506_492" src="https://github.com/user-attachments/assets/f18c2866-fc68-48f2-b9e4-05d1d2094b66" />

### Downward Position

<img width="640" height="480" alt="person_20260521_223642_659" src="https://github.com/user-attachments/assets/43e2aab7-e066-42fb-9fb8-def02467a3f8" />

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 2D Pose Estimation 데이터셋 구축 (2026년 5월)

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

2026년 5월에는 기존 데이터 수집 파이프라인을 활용하여 목 운동 중심의 데이터셋 설정을 어깨 운동 관련 4개 동작으로 변경했습니다. `config.json`을 수정하여 어깨 운동 레이블과 해당 이미지 폴더를 설정하고, 기존 이미지 수집 목표 개수를 유지했습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 5월
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 어깨 운동 관련 4개 동작을 대상으로 재활 운동 데이터셋 구성 확장
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)

## 2. 주요 구현 내용

### 데이터셋 설정 변경
- 기존 목 운동 5개 동작 클래스를 어깨 운동 4개 동작으로 변경
- `config.json`을 수정하여 어깨 운동 레이블 및 이미지 폴더 설정
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 자세 추정 및 데이터셋 생성 파이프라인에 변경된 설정 적용

### 자세 추정 및 데이터 수집
- YOLOv8n-pose를 활용하여 인체 자세 탐지 및 17개 관절 키포인트 추출
- 웹캠 영상에서 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표를 CSV로 저장

### 데이터셋 구축
- 어깨 운동 동작별 4개 이미지 폴더 설정
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 정규화된 관절 좌표와 동작 레이블 결합
- 향후 어깨 운동 동작 분류를 위한 구조화된 데이터셋 구성

## 3. 어깨 운동 동작 분류

`config.json`에 다음 4개 어깨 운동 동작을 정의했습니다.

| 레이블 | 동작 설명 | 폴더명 |
|---|---|---|
| `should_front` | 정면 자세 | `should_front_pose` |
| `should_up` | 위쪽 자세 | `should_up_pose` |
| `should_forward` | 전방 자세 | `should_forward_pose` |
| `should_down` | 아래쪽 자세 | `should_down_pose` |

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
   - 수집한 이미지를 어깨 운동 동작별 4개 폴더로 분류
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
    "should_front": "should_front_pose",
    "should_up": "should_up_pose",
    "should_forward": "should_forward_pose",
    "should_down": "should_down_pose"
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
  - `label`: 어깨 운동 동작 분류 레이블

## 6. 프로젝트 성과

- 기존 목 운동 5개 동작에서 어깨 운동 관련 4개 동작으로 데이터셋 설정 변경
- 프로그램 1회 실행 시 이미지 수집 목표 개수 200개 유지
- 기존 YOLOv8n-pose 기반 데이터 수집 및 레이블링 파이프라인을 어깨 운동에 적용
- 어깨 운동 동작 분류를 위한 구조화된 데이터 수집 환경 구성
- 향후 재활 운동 동작 인식 모델 개발을 위한 데이터셋 생성 과정 준비

## 7. 생성 사진

### 정면 자세

<img width="640" height="480" alt="person_20260521_223259_957" src="https://github.com/user-attachments/assets/7e7148de-cba2-43a2-b04f-566b47899d54" />

### 위쪽 자세

<img width="640" height="480" alt="person_20260521_223435_858" src="https://github.com/user-attachments/assets/a08728dc-1269-4dd3-a2d0-d37d348959a3" />

### 전방 자세

<img width="640" height="480" alt="person_20260521_223506_492" src="https://github.com/user-attachments/assets/f18c2866-fc68-48f2-b9e4-05d1d2094b66" />

### 아래쪽 자세

<img width="640" height="480" alt="person_20260521_223642_659" src="https://github.com/user-attachments/assets/43e2aab7-e066-42fb-9fb8-def02467a3f8" />

</details>
