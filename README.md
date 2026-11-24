<details>
<summary>ENG (English Version)</summary>

# Hemiplegia Rehabilitation Using AI-Based Pose Estimation

This repository documents a computer vision project that progressed from collecting pose data for hemiplegia rehabilitation exercises to recognizing poses from a live webcam feed. I collected **2,700 labeled frames across 15 poses** between March and July 2026 and used the combined dataset to train a pose classifier.

## 1. Rehabilitation Pose Datasets

Each monthly branch focuses on a different group of poses. The dataset pipeline uses YOLOv8n-pose to extract 17 body keypoints per person and saves the normalized coordinates and pose labels as CSV files.

### [1. March — Basic Postures](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

Built the initial data collection pipeline and collected **100 labeled frames** of sitting and standing.

### [2. April — Neck Movements](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

Collected **1,000 labeled frames** across five poses covering straight ahead, up, down, left, and right head positions.

### [3. May — Shoulder Movements](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

Collected **800 labeled frames** of four shoulder exercise poses with the elbows out to the sides, raised, forward, or down.

### [4. June — Arm Movements](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

Collected **400 labeled frames** of two poses with the arms overhead or the hands clasped in front.

### [5. July — Side Bends](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29)

Collected **400 labeled frames** of left and right trunk bends.

## 2. Real-Time Pose Recognition

### [6. Pose Recognition Application](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/Pose_Recognition)

Combined the five monthly datasets and trained a **Random Forest classifier** for all 15 poses. During recognition, YOLOv8n-pose extracts body keypoints, the classifier predicts the pose, and OpenCV displays the skeleton and pose name. The application accepts webcam, video, and image inputs.

### Demo Video

[Watch the real-time pose recognition demo on YouTube](https://www.youtube.com/watch?v=dA_wLsyu63g)

## Related Project

The project builds on an earlier webcam application that used YOLOv8 Pose and XGBoost to distinguish sitting from standing. [Human Pose Estimation — Sitting & Standing Classification](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 편마비 재활 운동을 위한 AI 기반 자세 추정 및 인식

편마비 재활 운동 자세의 데이터 수집부터 웹캠 실시간 인식까지 구현한 컴퓨터 비전 프로젝트입니다. 2026년 3~7월에 수집한 **15가지 자세의 레이블 프레임 2,700개**를 통합해 자세 분류 모델을 학습했습니다.

## 1. 재활 운동 자세 데이터셋

월별 브랜치에서 각 운동 유형의 데이터 수집 과정과 결과를 확인할 수 있습니다. YOLOv8n-pose로 사람마다 17개 관절을 검출하고, 정규화한 좌표와 자세 레이블을 CSV로 저장했습니다.

### [1. 3월 — 기본 자세](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/3%28March%29)

자세 데이터 수집 파이프라인을 구축하고, 앉기와 서기 두 자세의 프레임 **100개**를 수집했습니다.

### [2. 4월 — 목 운동](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/4%28April%29)

정면, 위, 아래, 왼쪽, 오른쪽을 보는 다섯 자세의 프레임 **1,000개**를 수집했습니다.

### [3. 5월 — 어깨 운동](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/5%28May%29)

팔꿈치를 옆으로 벌리기, 올리기, 앞으로 모으기, 내리기 네 자세의 프레임 **800개**를 수집했습니다.

### [4. 6월 — 팔 운동](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/6%28June%29)

머리 위로 팔을 올리거나 몸 앞에서 손을 모으는 두 자세의 프레임 **400개**를 수집했습니다.

### [5. 7월 — 몸통 좌우 기울이기](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/7%28July%29)

몸통을 왼쪽과 오른쪽으로 기울이는 두 자세의 프레임 **400개**를 수집했습니다.

## 2. 실시간 자세 인식

### [6. 실시간 자세 인식 프로그램](https://github.com/LutetiumCalciumLee/Hemiplegia_Rehabilitation_Using_AI-Based_Pose_Estimation_Project/tree/Pose_Recognition)

5개월치 데이터셋을 통합해 15가지 자세를 구분하는 **Random Forest 분류기**를 학습했습니다. 인식 단계에서는 YOLOv8n-pose로 관절을 검출하고, 분류기로 자세를 예측한 뒤 OpenCV 화면에 스켈레톤과 자세 이름을 표시합니다. 웹캠, 동영상, 이미지를 입력으로 사용할 수 있습니다.

### 시연 영상

[YouTube에서 실시간 자세 인식 시연 보기](https://www.youtube.com/watch?v=dA_wLsyu63g)

## 관련 프로젝트

이 프로젝트는 YOLOv8 Pose와 XGBoost를 사용해 웹캠에서 앉은 자세와 서 있는 자세를 구분한 이전 프로젝트를 바탕으로 진행했습니다. [인간 자세 추정 — 앉은 자세·서 있는 자세 분류](https://github.com/LutetiumCalciumLee/Studying_Computer_Vision/tree/Project_03)

</details>
