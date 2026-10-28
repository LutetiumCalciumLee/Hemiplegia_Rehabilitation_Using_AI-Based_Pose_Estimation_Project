
<details>
<summary>ENG (English Version)</summary>

# Hemiplegia Rehabilitation Using AI-Based Pose Estimation

This project focuses on constructing 2D human pose datasets for future rehabilitation exercise classification for patients with hemiplegia.

The project was developed based on my previous [**Human Pose Estimation Project**](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03), which used YOLOv8 Pose and XGBoost to classify sitting and standing postures.

Building on the pose estimation and data processing techniques explored in that project, I extended their application to rehabilitation exercise dataset construction.

From March to July 2026, I developed and used a data collection pipeline based on the pre-trained YOLOv8n-pose model. The pipeline extracts body keypoints from video frames, stores normalized coordinates, and generates labeled CSV datasets.

Throughout the five-month project, the dataset configuration was updated monthly to cover different rehabilitation exercise categories, including basic postures, neck movements, shoulder movements, arm movements, and left/right oblique postures.

## 1. Project Overview

- **Project Period:** March 2026 to July 2026 (5 Months)
- **Role:** Pose Estimation & Dataset Construction
- **Purpose:** Construct structured human pose datasets for future rehabilitation exercise classification.
- **Technologies:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **Model:** YOLOv8n-pose (Pre-trained)
- **Dataset Categories:** 15 movement labels across 5 monthly configurations
- **Based on:** [Human Pose Estimation: Sitting & Standing Classification](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)

### Project Background & Development

This project was developed based on the experience gained from the previous Human Pose Estimation project.

The previous project implemented:
- Real-time human pose estimation using YOLOv8 Pose.
- Extraction of 17 body keypoints from webcam frames.
- Construction of labeled datasets for sitting and standing postures.
- Training an XGBoost classifier for posture recognition.
- Real-time visualization of predicted postures.

The rehabilitation dataset project extended the application of pose estimation techniques:
- Applied human pose estimation to rehabilitation exercise data collection.
- Reused the concepts of keypoint extraction and labeled dataset construction.
- Configured multiple rehabilitation exercise categories using `config.json`.
- Generated structured CSV datasets for future exercise classification model development.


## 2. Main Responsibilities

### Pose Estimation
- Implemented 2D human pose estimation using a pre-trained YOLOv8n-pose model.
- Extracted 17 body keypoints from video frames.
- Visualized detected body keypoints and skeleton connections using OpenCV.
- Developed webcam-based pose data collection functionality.

### Data Collection
- Extracted normalized X and Y coordinates for 17 body keypoints.
- Processed every third video frame.
- Applied a detection confidence threshold greater than 0.75.
- Saved original images and corresponding keypoint coordinates.
- Implemented configurable data collection limits using `config.json`.

### Dataset Construction
- Organized captured images into posture-specific folders.
- Assigned movement labels based on image filenames and folder locations.
- Combined normalized keypoint coordinates with corresponding labels.
- Generated structured CSV datasets using Pandas.
- Adapted the data collection configuration for different rehabilitation exercise categories.

## 3. Monthly Project Development

The project was organized into five monthly GitHub branches, each focusing on a different set of rehabilitation movements.

| Period | Exercise Category | Classes |
|---|---|---|
| March 2026 | Basic Postures | 2 |
| April 2026 | Neck Movements | 5 |
| May 2026 | Shoulder Movements | 4 |
| June 2026 | Arm Movements | 2 |
| July 2026 | Oblique Postures | 2 | 


### [March: Basic Posture Dataset](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

- Constructed the initial rehabilitation pose estimation and data collection pipeline.
- Defined `sitting` and `standing` posture classes.
- Implemented keypoint extraction and CSV dataset generation.
- Generated a labeled dataset containing 100 samples in the provided March dataset.

**Classes:** Sitting, Standing

### [April: Neck Movement Dataset](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

- Changed the dataset configuration from basic postures to neck movements.
- Expanded the number of movement classes from 2 to 5.
- Increased the configurable image collection limit to 200 per execution.

**Classes:** Facing Forward, Looking Up, Looking Down, Looking Left, Looking Right

### [May: Shoulder Movement Dataset](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

- Changed the dataset configuration to shoulder movement categories.
- Defined 4 posture labels for shoulder movement data collection.
- Maintained the configurable image collection limit of 200 per execution.

**Classes:** Front Position, Upward Position, Forward Position, Downward Position

### [June: Arm Movement Dataset](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

- Changed the dataset configuration to arm movement categories.
- Defined 2 posture labels for raised and lowered arm positions.
- Reused the existing pose estimation and dataset generation pipeline.

**Classes:** Raised Arm Position, Lowered Arm Position

### [July: Oblique Posture Dataset](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29) 

- Changed the dataset configuration to left and right oblique postures.
- Defined 2 posture labels for left and right oblique positions.
- Maintained the existing data collection and labeling process.

**Classes:** Right Oblique Posture, Left Oblique Posture

## 4. Dataset Construction Pipeline

The same core data processing pipeline was used throughout the project.

1. **Pose Detection**
   - Capture video frames using OpenCV.
   - Detect human poses using the pre-trained YOLOv8n-pose model.
   - Extract 17 body keypoints.

2. **Keypoint Extraction**
   - Extract normalized X and Y coordinates.
   - Store keypoint coordinates as `x0, y0, ..., x16, y16`.

3. **Data Collection**
   - Save original images from webcam frames.
   - Store image filenames and keypoint coordinates in `keypoints.csv`.
   - Apply the configurable image collection limit.

4. **Data Labeling**
   - Organize collected images into posture-specific folders.
   - Assign movement labels based on image filenames and folder locations.
   - Exclude samples without corresponding labels.

5. **Dataset Generation**
   - Combine keypoint coordinates with movement labels.
   - Generate the final structured dataset using Pandas.
   - Export labeled data to `dataset.csv`.

## 5. Dataset Structure & Project Files

### Dataset Structure

Two CSV files are used in the dataset construction process.

| File | Columns | Description |
|---|---|---|
| `keypoints.csv` | 35 | Image filename and 34 coordinate values from 17 body keypoints |
| `dataset.csv` | 36 | Keypoint data with an additional movement label |

### keypoints.csv

- `image_name`: Original image filename
- `x0, y0, ..., x16, y16`: Normalized X and Y coordinates of 17 body keypoints

### dataset.csv

- `image_name`: Original image filename
- `x0, y0, ..., x16, y16`: Normalized body keypoint coordinates
- `label`: Corresponding rehabilitation movement label

### Main Project Files

- **1.2D_Pose_Estimation_test.py**
  - Tests human pose estimation using a video file.
  - Visualizes detected body keypoints and skeleton connections.

- **2.2D_Pose_Save.py**
  - Captures webcam frames and extracts body keypoints.
  - Saves original images and normalized coordinates to CSV.

- **3.2D_Pose_Dataset_Save.py**
  - Assigns posture labels based on classified image folders.
  - Generates the final labeled dataset.

- **config.json**
  - Defines movement categories and image collection limits.
  - Allows the data collection pipeline to be reused for different exercise categories.

- **Models/yolov8n-pose.pt**
  - Pre-trained YOLOv8n pose estimation model.

## 6. Project Outcomes

- Extended pose estimation and dataset processing techniques from the previous [Human Pose Estimation Project](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03) to rehabilitation exercise data collection.
- Developed a reusable 2D human pose data collection pipeline using YOLOv8n-pose.
- Implemented extraction and CSV storage of 17 body keypoints.
- Constructed a dataset generation process combining normalized coordinates and movement labels.
- Configured 15 movement labels across 5 monthly dataset configurations.
- Adapted the existing pipeline to different rehabilitation exercise categories without requiring a new pose estimation model.
- Organized monthly implementation details and generated images into separate GitHub branches.
- Prepared structured pose datasets for future rehabilitation exercise classification model development.

**Project Scope:** This project focuses on pose data collection, labeling, and dataset construction. Training and evaluating a rehabilitation exercise classification model are outside the scope of the documented implementation.

## 7. Generated Images

The generated images and monthly dataset results are documented in the corresponding GitHub branches.

### March: Basic Postures

[View March Project & Images](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

### April: Neck Movements

[View April Project & Images](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

### May: Shoulder Movements

[View May Project & Images](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

### June: Arm Movements

[View June Project & Images](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

### July: Oblique Postures

[View July Project & Images](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29)

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 편마비 환자 재활 운동을 위한 AI 기반 Pose Estimation 데이터셋 구축

이 프로젝트는 편마비 환자의 재활 운동 동작을 향후 분류할 수 있도록 2D 인체 자세 데이터셋을 구축하는 프로젝트입니다.

본 프로젝트는 이전에 진행했던 [**Human Pose Estimation 프로젝트**](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)를 기반으로 개발했습니다. 해당 프로젝트에서는 YOLOv8 Pose와 XGBoost를 활용하여 앉은 자세와 서 있는 자세를 분류하는 시스템을 구현했습니다.

이 과정에서 학습한 자세 추정 및 데이터 처리 기술을 바탕으로, 인체 자세 추정 기술의 활용 범위를 재활 운동 데이터셋 구축으로 확장했습니다.

2026년 3월부터 7월까지 사전 학습된 YOLOv8n-pose 모델을 활용하여 영상에서 인체 관절 좌표를 추출하고, 정규화된 좌표와 동작별 레이블이 포함된 CSV 데이터셋을 생성하는 파이프라인을 구축했습니다.

5개월 동안 기본 자세, 목 운동, 어깨 운동, 팔 운동, 좌우 기울임 자세 등 다양한 재활 운동 동작을 대상으로 매월 데이터셋 설정을 변경하여 작업을 수행했습니다.

## 1. 프로젝트 개요

- **진행 기간:** 2026년 3월 ~ 2026년 7월 (5개월)
- **담당 역할:** 자세 추정 및 데이터셋 구축
- **목적:** 향후 재활 운동 동작 분류를 위한 구조화된 인체 자세 데이터셋 구축
- **사용 기술:** Python, YOLOv8n-pose, OpenCV, Pandas, Tkinter, JSON, CSV
- **사용 모델:** YOLOv8n-pose (사전 학습 모델)
- **데이터셋 구성:** 5개 월별 설정에 걸쳐 총 15개 동작 레이블 정의
- **기반 프로젝트:** [Human Pose Estimation: Sitting & Standing Classification](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)

### 프로젝트 개발 배경 및 확장 과정

본 프로젝트는 이전에 진행한 Human Pose Estimation 프로젝트에서 습득한 기술과 구현 경험을 바탕으로 개발했습니다.

기존 프로젝트에서 구현한 내용은 다음과 같습니다.

- YOLOv8 Pose를 활용한 실시간 인체 자세 추정
- 웹캠 영상에서 17개 인체 관절 키포인트 추출
- 앉은 자세와 서 있는 자세의 레이블 데이터셋 구축
- XGBoost 분류 모델 학습을 통한 자세 분류
- 실시간 자세 분류 결과 시각화

이후 해당 기술을 재활 운동 데이터셋 구축에 활용할 수 있도록 확장했습니다.

- 인체 자세 추정 기술을 재활 운동 데이터 수집에 적용
- 관절 키포인트 추출 및 레이블 데이터셋 구축 방식 활용
- `config.json`을 활용한 다양한 재활 운동 동작 클래스 설정
- 향후 재활 운동 동작 분류 모델 학습을 위한 구조화된 CSV 데이터셋 생성


## 2. 주요 담당 업무

### 자세 추정 (Pose Estimation)
- 사전 학습된 YOLOv8n-pose 모델을 활용한 2D 인체 자세 추정 구현
- 영상 프레임에서 17개 인체 관절 키포인트 추출
- OpenCV를 활용한 관절 좌표 및 스켈레톤 시각화
- 웹캠 기반 자세 데이터 수집 기능 구현

### 데이터 수집 (Data Collection)
- 17개 관절 키포인트의 정규화된 X, Y 좌표 추출
- 영상 프레임을 3프레임마다 처리
- 객체 탐지 신뢰도 0.75 초과 조건 적용
- 원본 이미지와 해당 관절 좌표 저장
- `config.json`을 활용한 이미지 수집 목표 개수 설정

### 데이터셋 구축 (Dataset Construction)
- 수집한 이미지를 자세별 폴더로 분류
- 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
- 정규화된 관절 좌표와 해당 동작 레이블 결합
- Pandas를 활용한 구조화된 CSV 데이터셋 생성
- 재활 운동 동작별 데이터 수집 설정 변경 및 적용

## 3. 월별 프로젝트 진행 내용

프로젝트는 총 5개의 월별 GitHub 브랜치로 구분하여 관리했습니다.

| 진행 기간 | 운동 유형 | 동작 클래스 |
|---|---|---|
| 2026년 3월 | 기본 자세 | 2개 |
| 2026년 4월 | 목 운동 | 5개 |
| 2026년 5월 | 어깨 운동 | 4개 |
| 2026년 6월 | 팔 운동 | 2개 |
| 2026년 7월 | 좌우 기울임 자세 | 2개 |

### [3월: 기본 자세 데이터셋 구축](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

- 재활 운동용 초기 자세 추정 및 데이터 수집 파이프라인 구현
- 앉기(`sitting`)와 서 있기(`standing`) 동작 클래스 정의
- 관절 키포인트 추출 및 CSV 데이터셋 생성 기능 구현
- 첨부된 3월 데이터 기준 총 100개 샘플로 구성된 레이블 데이터셋 생성

**동작 분류:** 앉기, 서 있기

### [4월: 목 운동 데이터셋 구축](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

- 기본 자세에서 목 운동 동작으로 데이터셋 설정 변경
- 동작 클래스를 기존 2개에서 5개로 확장
- 프로그램 1회 실행 시 이미지 수집 목표 개수를 200개로 변경

**동작 분류:** 정면, 위쪽, 아래쪽, 왼쪽, 오른쪽

### [5월: 어깨 운동 데이터셋 구축](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

- 목 운동에서 어깨 운동 동작으로 데이터셋 설정 변경
- 어깨 운동 관련 4개 자세 레이블 정의
- 이미지 수집 목표 개수 200개 유지

**동작 분류:** 정면 자세, 위쪽 자세, 전방 자세, 아래쪽 자세

### [6월: 팔 운동 데이터셋 구축](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

- 어깨 운동에서 팔 운동 동작으로 데이터셋 설정 변경
- 팔 올리기와 팔 내리기 2개 자세 레이블 정의
- 기존 자세 추정 및 데이터셋 생성 파이프라인 재사용

**동작 분류:** 팔 올리기, 팔 내리기

### [7월: 좌우 기울임 자세 데이터셋 구축](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29)

- 팔 운동에서 좌우 기울임 자세로 데이터셋 설정 변경
- 우측 및 좌측 기울임 자세 2개 레이블 정의
- 기존 데이터 수집 및 레이블링 과정 유지

**동작 분류:** 우측 기울임 자세, 좌측 기울임 자세

## 4. 데이터셋 구축 파이프라인

프로젝트 기간 동안 동일한 핵심 데이터 처리 파이프라인을 활용했습니다.

1. **자세 탐지 (Pose Detection)**
   - OpenCV를 활용한 영상 프레임 입력
   - 사전 학습된 YOLOv8n-pose 모델을 이용한 인체 자세 탐지
   - 영상 프레임에서 17개 관절 키포인트 추출

2. **키포인트 추출 (Keypoint Extraction)**
   - 정규화된 X, Y 좌표 추출
   - `x0, y0, ..., x16, y16` 형태로 관절 좌표 구성

3. **데이터 수집 (Data Collection)**
   - 웹캠에서 원본 이미지 수집
   - 이미지 파일명과 관절 좌표를 `keypoints.csv`에 저장
   - 설정된 이미지 수집 목표 개수 적용

4. **데이터 레이블링 (Data Labeling)**
   - 수집한 이미지를 동작별 폴더로 분류
   - 이미지 파일명과 폴더 위치를 기준으로 동작 레이블 부여
   - 레이블이 없는 데이터 제외

5. **데이터셋 생성 (Dataset Generation)**
   - 정규화된 관절 좌표와 동작 레이블 결합
   - Pandas를 활용한 구조화된 학습용 데이터셋 생성
   - 최종 레이블 데이터를 `dataset.csv`로 저장

## 5. 데이터셋 구조 및 프로젝트 파일

### 데이터셋 구조

데이터셋 구축 과정에서 두 종류의 CSV 파일을 사용했습니다.

| 파일명 | 열 개수 | 구성 |
|---|---|---|
| `keypoints.csv` | 35개 | 이미지 파일명과 17개 관절의 좌표 34개 |
| `dataset.csv` | 36개 | 관절 좌표 데이터와 동작 레이블 |

### keypoints.csv

- `image_name`: 원본 이미지 파일명
- `x0, y0, ..., x16, y16`: 17개 관절의 정규화된 X, Y 좌표

### dataset.csv

- `image_name`: 원본 이미지 파일명
- `x0, y0, ..., x16, y16`: 정규화된 관절 좌표
- `label`: 해당 재활 운동 동작의 분류 레이블

### 주요 프로젝트 파일

- **1.2D_Pose_Estimation_test.py**
  - 동영상 파일을 이용한 인체 자세 추정 기능 테스트
  - 관절 키포인트 및 스켈레톤 시각화

- **2.2D_Pose_Save.py**
  - 웹캠 영상에서 인체 관절 키포인트 추출
  - 원본 이미지 및 정규화된 관절 좌표를 CSV로 저장

- **3.2D_Pose_Dataset_Save.py**
  - 분류된 이미지 폴더를 기준으로 동작 레이블 부여
  - 최종 레이블 데이터셋 생성

- **config.json**
  - 운동 동작 클래스 및 이미지 수집 목표 개수 설정
  - 월별 운동 유형 변경 시 기존 데이터 수집 프로그램 재사용 지원

- **Models/yolov8n-pose.pt**
  - 사전 학습된 YOLOv8n 자세 추정 모델

## 6. 프로젝트 성과

- 기존 [Human Pose Estimation 프로젝트](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)에서 학습한 자세 추정 및 데이터 처리 기술을 재활 운동 데이터 수집에 활용
- YOLOv8n-pose를 활용한 재사용 가능한 2D 인체 자세 데이터 수집 파이프라인 구현
- 17개 관절 키포인트 추출 및 CSV 저장 자동화
- 정규화된 관절 좌표와 동작 레이블을 결합하는 데이터셋 생성 과정 구현
- 5개월에 걸쳐 총 15개 동작 레이블을 대상으로 데이터셋 설정 구성
- 자세 추정 모델을 새로 개발하지 않고 기존 파이프라인을 다양한 재활 운동 동작에 적용
- 월별 구현 내용과 생성 사진을 GitHub 브랜치별로 구분하여 관리
- 향후 재활 운동 동작 분류 모델 개발에 활용할 수 있는 구조화된 자세 데이터셋 준비

**구현 범위:** 본 프로젝트는 자세 데이터 수집, 레이블링 및 데이터셋 구축에 중점을 두었습니다. 재활 운동 분류 모델의 학습 및 성능 평가는 현재 정리된 구현 범위에 포함되지 않습니다.

## 7. 생성 사진

각 월에 생성한 자세 추정 이미지와 데이터셋 구축 결과는 해당 GitHub 브랜치에서 확인할 수 있습니다.

### 3월: 기본 자세

[3월 프로젝트 및 생성 사진 보기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

### 4월: 목 운동

[4월 프로젝트 및 생성 사진 보기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

### 5월: 어깨 운동

[5월 프로젝트 및 생성 사진 보기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

### 6월: 팔 운동

[6월 프로젝트 및 생성 사진 보기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

### 7월: 좌우 기울임 자세

[7월 프로젝트 및 생성 사진 보기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29)

</details>
