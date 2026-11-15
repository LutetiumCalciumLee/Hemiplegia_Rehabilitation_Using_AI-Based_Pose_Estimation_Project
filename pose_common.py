"""학습과 인식에서 공유하는 COCO 17개 관절 전처리."""

from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(17) for axis in ("x", "y")]
FEATURE_VERSION = "shoulder_center_shoulder_width_visibility_v2"
DEFAULT_CLASSIFIER = BASE_DIR / "Models" / "pose_classifier.joblib"
DEFAULT_POSE_MODEL = BASE_DIR / "Models" / "yolov8n-pose.pt"


def pose_features(points):
    """(N,17,2) xyn -> 상대 좌표 34개 + 관절 존재 여부 17개.

    수집 코드의 keypoints.xyn과 같은 입력을 사용한다. (0,0)은 미검출이다.
    좌우 반전은 하지 않는다. left/right 클래스의 의미가 바뀌기 때문이다.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (17, 2):
        raise ValueError("관절 좌표는 (사람 수, 17, 2) 모양이어야 합니다.")
    if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError("관절 좌표는 0~1 사이의 유한한 숫자여야 합니다.")
    visible = np.any(points != 0, axis=2)
    if (visible.sum(axis=1) < 5).any():
        raise ValueError("검출된 관절이 5개 미만인 사람이 있습니다.")
    features = []
    for pose, mask in zip(points, visible):
        shoulders_ok = bool(mask[5] and mask[6])
        center = pose[[5, 6]].mean(axis=0) if shoulders_ok else pose[mask].mean(axis=0)
        # 하체가 프레임 밖으로 나가도 상체 특징의 기준 크기가 바뀌지 않게 한다.
        scale = float(np.linalg.norm(pose[5] - pose[6])) if shoulders_ok else 0.0
        if scale < 1e-4:
            scale = max(float(np.ptp(pose[mask], axis=0).max()), 1e-4)
        relative = (pose - center) / scale
        relative[~mask] = 0
        features.append(np.concatenate((relative.ravel(), mask.astype(np.float32))))
    return np.asarray(features, dtype=np.float32).reshape(-1, 51)


def augment_training_poses(points, seed=42):
    """학습 구간에서만 하체 미검출 및 작은 검출 오차를 모사한다.

    좌우/상하 반전은 라벨 의미가 바뀌므로 사용하지 않는다.
    같은 샘플의 변형은 반드시 원본과 같은 학습 구간에 둔다.
    """
    points = np.asarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    shoulder_width = np.linalg.norm(points[:, 5] - points[:, 6], axis=1)
    center = points[:, [5, 6]].mean(axis=1, keepdims=True)
    visible = np.any(points != 0, axis=2)
    batches = []
    for noisy in (False, True):
        transformed = points.copy()
        if noisy:
            # 화면 비율과 자세 검출 좌표의 작은 차이를 모사한다.
            stretch = rng.uniform(0.85, 1.15, size=(len(points), 1, 2))
            noise = rng.normal(size=points.shape) * shoulder_width[:, None, None] * 0.025
            transformed = np.clip((points - center) * stretch + center + noise, 0, 1)
            transformed[~visible] = 0
        for missing_from in (17, 13, 11):
            variant = transformed.copy()
            variant[:, missing_from:] = 0
            batches.append(variant)
    return np.concatenate(batches, axis=0).astype(np.float32)


def apply_digital_zoom(image, factor):
    """기존 수집 코드와 동일한 중앙 크롭. factor=1이면 원본."""
    if factor < 1:
        raise ValueError("zoom은 1 이상이어야 합니다.")
    if factor == 1:
        return image
    import cv2

    height, width = image.shape[:2]
    crop_w, crop_h = max(1, int(width / factor)), max(1, int(height / factor))
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return cv2.resize(image[top : top + crop_h, left : left + crop_w], (width, height))
