"""3~7월 dataset.csv를 합쳐 자세 분류기를 학습한다.

실행: python train_pose.py
다른 경로: python train_pose.py --data "D:/data/pose_img" "D:/data/another"
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pose_common import (
    BASE_DIR, DEFAULT_CLASSIFIER, FEATURE_COLUMNS, FEATURE_VERSION,
    augment_training_poses, pose_features,
)


def find_csv(path):
    path = Path(path).expanduser().resolve()
    candidates = [path] if path.is_file() else [path / "dataset.csv", path / "pose_img" / "dataset.csv"]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".csv":
            return candidate
    raise ValueError(f"dataset.csv를 찾을 수 없습니다: {path}")


def load_datasets(paths, allowed_labels):
    frames, sources, visited = [], [], set()
    for path in paths:
        csv_path = find_csv(path)
        if csv_path in visited:
            continue
        visited.add(csv_path)
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        required = ["image_name", *FEATURE_COLUMNS, "label"]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{csv_path}: 누락된 열 {missing}")
        frame = frame[required].copy()
        if frame.empty or frame[["image_name", "label"]].isna().any().any():
            raise ValueError(f"{csv_path}: 비어 있거나 이미지명/라벨이 누락되었습니다.")
        frame["label"] = frame["label"].astype(str).str.strip()
        unknown = sorted(set(frame["label"]) - set(allowed_labels))
        if unknown:
            raise ValueError(f"{csv_path}: config.json에 없는 라벨 {unknown}")
        if frame["image_name"].duplicated().any():
            raise ValueError(f"{csv_path}: 중복 image_name을 먼저 정리해 주세요.")
        # 수집 시각으로 분리하므로 수집 코드의 파일명 형식을 확인한다.
        times = pd.to_datetime(
            frame["image_name"].astype(str).str.extract(
                r"^person_(\d{8}_\d{6}_\d{3})\.[^.]+$", expand=False
            ),
            format="%Y%m%d_%H%M%S_%f", errors="coerce",
        )
        if times.isna().any():
            raise ValueError(f"{csv_path}: person_YYYYMMDD_HHMMSS_mmm.jpg 형식의 이미지명이 필요합니다.")
        frame["captured_at"] = times
        try:
            frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
            pose_features(frame[FEATURE_COLUMNS].to_numpy().reshape(-1, 17, 2))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{csv_path}: 잘못된 관절 데이터: {exc}") from exc
        frame["source"] = str(csv_path)
        sources.append({"csv": str(csv_path), "rows": len(frame)})
        frames.append(frame)
    if not frames:
        raise ValueError("학습할 CSV가 없습니다.")
    merged = pd.concat(frames, ignore_index=True)
    # 같은 좌표에 서로 다른 정답이 붙어 있으면 학습하지 않는다.
    conflicts = merged.groupby(FEATURE_COLUMNS, dropna=False)["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("동일한 관절 좌표에 서로 다른 라벨이 있습니다. 라벨을 확인해 주세요.")
    before = len(merged)
    merged = merged.drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if merged["label"].nunique() < 2:
        raise ValueError("최소 두 가지 자세가 필요합니다.")
    return merged, sources, before - len(merged)


def temporal_split(frame, validation_fraction=0.2, gap_fraction=0.1):
    """각 원본 CSV/라벨의 마지막 구간을 검증하고 경계 프레임은 제외한다.

    같은 촬영 안에서 나누는 보조 검증이며 독립 촬영 성능을 의미하지 않는다.
    """
    train, validation, gap = [], [], []
    for (source, label), group in frame.groupby(["source", "label"], sort=True):
        indices = group.sort_values(["captured_at", "image_name"]).index.tolist()
        n_validation = max(1, math.ceil(len(indices) * validation_fraction))
        n_gap = math.ceil(len(indices) * gap_fraction)
        boundary = len(indices) - n_validation
        if boundary - n_gap < 2:
            raise ValueError(f"{source} / {label}: 분할할 데이터가 부족합니다 ({len(indices)}개).")
        train.extend(indices[: boundary - n_gap])
        gap.extend(indices[boundary - n_gap : boundary])
        validation.extend(indices[boundary:])
    return np.array(train, dtype=int), np.array(validation, dtype=int), np.array(gap, dtype=int)


def main(argv=None):
    parser = argparse.ArgumentParser(description="17개 관절 좌표로 15가지 자세 학습")
    parser.add_argument("--data", nargs="+", type=Path, help="월별 폴더, pose_img 폴더 또는 dataset.csv")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--report-dir", type=Path, default=BASE_DIR / "training_results")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--gap-fraction", type=float, default=0.1)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1 or not 0 <= args.gap_fraction < 1:
        parser.error("검증 비율은 0 초과 1 미만, 경계 제외 비율은 0 이상 1 미만이어야 합니다.")
    if args.validation_fraction + args.gap_fraction >= 1 or args.trees < 1:
        parser.error("검증+경계 제외 비율은 1 미만, trees는 1 이상이어야 합니다.")
    paths = args.data or [BASE_DIR.parent / f"pose_img_{month}월_재혁" for month in range(3, 8)]
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    allowed_labels = config.get("pose", {})
    if not allowed_labels:
        raise ValueError("config.json에 pose 라벨 목록이 없습니다.")
    frame, sources, removed = load_datasets(paths, allowed_labels)
    train, validation, gap = temporal_split(frame, args.validation_fraction, args.gap_fraction)
    points = frame[FEATURE_COLUMNS].to_numpy().reshape(-1, 17, 2)
    features = pose_features(points)
    labels = frame["label"].to_numpy()
    print(f"총 {len(frame)}개 / {frame['label'].nunique()}개 클래스 (중복 제거 {removed}개)")
    print(f"학습 {len(train)}개 / 검증 {len(validation)}개 / 경계 제외 {len(gap)}개", flush=True)
    classifier = RandomForestClassifier(
        n_estimators=args.trees, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
    )
    augmented_points = augment_training_poses(points[train], seed=args.seed)
    augmented_labels = np.tile(labels[train], len(augmented_points) // len(train))
    print(f"하체 미검출/좌표 오차 증강 후 학습 샘플: {len(augmented_points)}개", flush=True)
    classifier.fit(pose_features(augmented_points), augmented_labels)
    predicted = classifier.predict(features[validation])
    probabilities = classifier.predict_proba(features[validation])
    accuracy = float(accuracy_score(labels[validation], predicted))
    classes = classifier.classes_.tolist()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier", "sklearn_version": sklearn.__version__,
        "feature_version": FEATURE_VERSION, "sources": sources,
        "augmented_train_rows": len(augmented_points),
        "augmentation": "training_only_full_no_legs_no_lower_body_with_small_noise_and_xy_stretch",
        "rows": len(frame), "duplicate_rows_removed": removed,
        "train_rows": len(train), "validation_rows": len(validation), "gap_rows": len(gap),
        "validation_fraction": args.validation_fraction, "gap_fraction": args.gap_fraction,
        "split": "per_source_per_label_chronological_tail_with_gap",
        "limitation": "Validation uses the same recording sessions. Not independent-person/session accuracy.",
        "seed": args.seed, "trees": args.trees, "classes": classes,
        "class_counts": {k: int(v) for k, v in frame["label"].value_counts().items()},
        "accuracy": accuracy,
        "classification_report": classification_report(
            labels[validation], predicted, labels=classes, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels[validation], predicted, labels=classes).tolist(),
    }
    # 실제 웹캠 성능과는 별개로, 가림에 대한 회귀 검사를 함께 남긴다.
    report["synthetic_visibility_checks"] = {}
    for name, missing_from in (("full", 17), ("no_legs", 13), ("no_lower_body", 11)):
        modified = points[validation].copy()
        modified[:, missing_from:] = 0
        scores = classifier.predict_proba(pose_features(modified))
        best = scores.argmax(axis=1)
        predicted_modified = classifier.classes_[best]
        accepted = scores.max(axis=1) >= 0.55
        arm_rows = labels[validation] == "arm_up"
        report["synthetic_visibility_checks"][name] = {
            "note": "Synthetic masking of held-out coordinates; not independent webcam evaluation.",
            "accuracy": float(np.mean(predicted_modified == labels[validation])),
            "unknown_rate_at_0.55": float(np.mean(~accepted)),
            "arm_up_rows": int(arm_rows.sum()),
            "arm_up_accepted_correct": int(np.sum(
                arm_rows & accepted & (predicted_modified == labels[validation])
            )),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "classifier": classifier, "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS, "sklearn_version": sklearn.__version__,
        "classes": classes, "report": report,
    }, args.output, compress=3)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = frame[["source", "image_name", "label"]].copy()
    manifest["split"] = "gap"
    manifest.loc[train, "split"] = "train"
    manifest.loc[validation, "split"] = "validation"
    manifest.to_csv(args.report_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    predictions = manifest.loc[validation].copy()
    predictions["predicted"] = predicted
    predictions["score"] = probabilities.max(axis=1)
    predictions.to_csv(args.report_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    print(classification_report(labels[validation], predicted, labels=classes, zero_division=0))
    print(f"검증 정확도: {accuracy:.2%} (같은 촬영 구간 내부 검증)")
    print(f"저장된 분류기: {args.output.resolve()}")
    print(f"결과: {args.report_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"학습 실패: {exc}", file=sys.stderr)
        sys.exit(1)
