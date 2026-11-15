"""웹캠·동영상·이미지에서 사람별 자세 인식. q 또는 ESC로 종료."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from pose_common import (
    DEFAULT_CLASSIFIER, DEFAULT_POSE_MODEL, FEATURE_VERSION, apply_digital_zoom, pose_features,
)

POSE_DISPLAY_NAMES = {
    "sitting": "Sitting",
    "standing": "Standing",
    "neck_front": "Looking Straight Ahead",
    "neck_up": "Looking Up",
    "neck_down": "Looking Down",
    "neck_left": "Looking Left",
    "neck_right": "Looking Right",
    "should_front": "Elbows Out to the Sides",
    "should_up": "Elbows Raised",
    "should_forward": "Elbows Forward",
    "should_down": "Elbows Down",
    "arm_up": "Arms Overhead",
    "arm_down": "Hands Clasped in Front",
    "obl_right": "Bending to the Right",
    "obl_left": "Bending to the Left",
    "unknown": "Unknown",
}

DEFAULT_CAMERA_INDEX = 1


def pose_display_name(label):
    """학습용 라벨/폴더명을 사람이 읽는 자세 이름으로 바꾼다."""
    label = label.removesuffix("_pose")
    return POSE_DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def load_classifier(path):
    if not path.is_file():
        raise ValueError(f"분류기가 없습니다. 먼저 python train_pose.py 실행: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("전처리 버전이 다릅니다. 현재 train_pose.py로 다시 학습해 주세요.")
    return bundle["classifier"]


def classify_result(result, classifier, threshold, keypoint_confidence=0.5):
    """사람별 결과. score는 보정된 확률이 아닌 분류기 점수다."""
    if result.keypoints is None or result.boxes is None:
        return []
    points = result.keypoints.xyn.cpu().numpy().copy()
    # 기존 YOLOv8 수집 코드의 미검출 관절(0,0) 처리와 맞춘다.
    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf.cpu().numpy()
        points[confidence < keypoint_confidence] = 0
    boxes = result.boxes.xyxy.cpu().numpy()
    records, usable, indices = [], [], []
    for index, (pose, box) in enumerate(zip(points, boxes)):
        visible = np.any(pose != 0, axis=1)
        count = int(visible.sum())
        records.append({
            "person": index, "label": "unknown", "display_label": "Unknown",
            "score": None, "box": box.tolist(),
            "visible_keypoints": count, "candidate_label": None,
            "candidate_display_label": None, "alternatives": [],
            "reason": "too_few_keypoints",
        })
        if count < 5:
            continue
        if not visible[5] or not visible[6]:
            records[index]["reason"] = "shoulders_not_visible"
            continue
        if (not np.isfinite(pose).all() or (pose < 0).any() or (pose > 1).any()
                or np.linalg.norm(pose[5] - pose[6]) < 1e-4):
            records[index]["reason"] = "invalid_keypoints"
            continue
        usable.append(pose)
        indices.append(index)
    if usable:
        scores = classifier.predict_proba(pose_features(np.asarray(usable)))
        for index, score in zip(indices, scores):
            best = int(np.argmax(score))
            value = float(score[best])
            candidate = str(classifier.classes_[best])
            label = candidate if value >= threshold else "unknown"
            records[index].update(
                label=label, display_label=pose_display_name(label),
                score=value, candidate_label=candidate,
                candidate_display_label=pose_display_name(candidate),
                reason="recognized" if value >= threshold else "low_score",
                alternatives=[{
                    "label": str(classifier.classes_[i]),
                    "display_label": pose_display_name(str(classifier.classes_[i])),
                    "score": float(score[i]),
                } for i in np.argsort(score)[::-1][:3]],
            )
    return records


def annotate(result, records, debug=False, threshold=0.55, zoom=1.0, *, mirror=False):
    import cv2

    frame = result.plot(labels=False, boxes=False)
    if mirror:
        frame = cv2.flip(frame, 1)
    for record in records:
        x1, y1, x2, y2 = [int(value) for value in record["box"]]
        if mirror:
            last_x = frame.shape[1] - 1
            x1, x2 = last_x - x2, last_x - x1
        color = (0, 200, 0) if record["label"] != "unknown" else (0, 180, 255)
        text = pose_display_name(record["label"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (max(0, x1), max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    # 대표 인물의 자세를 화면 왼쪽 위에 크게 표시한다.
    # 여러 사람이 있으면 화면에서 가장 크게 잡힌 사람을 대표로 사용한다.
    primary = max(records, key=lambda r: (r["box"][2] - r["box"][0]) *
                  (r["box"][3] - r["box"][1]), default=None)
    title = pose_display_name(primary["label"]) if primary else "No Person Detected"
    color = (0, 200, 0) if primary and primary["label"] != "unknown" else (0, 180, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.95, max(0.35, frame.shape[1] / 640))
    (text_width, text_height), baseline = cv2.getTextSize(title, font, scale, 2)
    cv2.rectangle(frame, (8, 8), (text_width + 28, text_height + baseline + 28), (25, 25, 25), -1)
    cv2.putText(frame, title, (18, text_height + 18), font, scale, color, 2, cv2.LINE_AA)
    details = []
    if primary and primary["label"] == "unknown":
        if primary["reason"] == "low_score":
            details.append(f"Closest: {primary['candidate_display_label']}  score={primary['score']:.2f}")
        elif primary["reason"] == "shoulders_not_visible":
            details.append("Show both shoulders to the camera")
        elif primary["reason"] == "too_few_keypoints":
            details.append("Step back: too few body points visible")
        else:
            details.append("Move into view of the camera")
    if debug and primary:
        details.append(f"Points: {primary['visible_keypoints']}/17   cutoff: {threshold:.2f}   zoom: {zoom:.1f}")
        if primary["alternatives"]:
            details.append(" / ".join(f"{item['display_label']}: {item['score']:.2f}"
                                      for item in primary["alternatives"]))
    y = text_height + baseline + 48
    for text in details:
        if y + 5 >= frame.shape[0]:
            break
        detail_scale = min(0.5, max(0.25, (frame.shape[1] - 30) /
                                    max(1, cv2.getTextSize(text, font, 1.0, 1)[0][0])))
        (width, height), base = cv2.getTextSize(text, font, detail_scale, 1)
        cv2.rectangle(frame, (8, y - height - 4), (width + 24, y + base + 4), (25, 25, 25), -1)
        cv2.putText(frame, text, (16, y), font, detail_scale, (230, 230, 230), 1, cv2.LINE_AA)
        y += height + base + 14
    return frame


def fit_fullscreen_preview(frame, width, height):
    """전체 화면에 맞춰 원본 영상 비율을 유지한다."""
    import cv2

    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    image_width = max(1, round(frame_width * scale))
    image_height = max(1, round(frame_height * scale))
    resized = cv2.resize(frame, (image_width, image_height))
    left = (width - image_width) // 2
    top = (height - image_height) // 2
    return cv2.copyMakeBorder(
        resized, top, height - image_height - top,
        left, width - image_width - left, cv2.BORDER_CONSTANT, value=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 Pose + 학습된 자세 분류기")
    parser.add_argument("--source", default=str(DEFAULT_CAMERA_INDEX),
                        help="웹캠 번호, 동영상 또는 이미지 경로 (기본: 1, 외장 웹캠)")
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--threshold", type=float, default=0.55, help="미만이면 unknown (기본: 0.55)")
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--keypoint-confidence", type=float, default=0.5, help="관절 검출 최소 점수 (기본: 0.5)")
    parser.add_argument("--zoom", type=float, default=1.0, help="중앙 확대 배율 (기본: 1, 확대 안 함)")
    parser.add_argument("--debug", action="store_true", help="관절 수와 상위 자세 후보 점수 표시")
    parser.add_argument("--device", default="cpu", help="cpu 또는 CUDA GPU 번호 (예: 0)")
    parser.add_argument("--output", type=Path, help="인식 이미지 또는 mp4 저장 경로")
    parser.add_argument("--json-output", type=Path, help="프레임별 인식 결과 JSONL 저장 경로")
    parser.add_argument("--no-display", action="store_true", help="창 없이 실행")
    args = parser.parse_args(argv)
    if (not 0 <= args.threshold <= 1 or not 0 < args.detection_confidence <= 1
            or not 0 <= args.keypoint_confidence <= 1 or not np.isfinite(args.zoom) or args.zoom < 1):
        parser.error("threshold/keypoint-confidence는 0~1, detection-confidence는 0 초과 1 이하, zoom은 1 이상입니다.")
    classifier = load_classifier(args.classifier)
    if not args.pose_model.is_file():
        raise ValueError(f"YOLO 포즈 모델이 없습니다: {args.pose_model}")
    source_path = Path(args.source)
    camera = args.source.isdecimal()
    if not camera and not source_path.is_file():
        raise ValueError(f"입력 파일이 없습니다: {source_path}")
    still_image = not camera and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.output:
        if still_image and args.output.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            parser.error("이미지 출력은 jpg/png/bmp/webp 확장자를 사용해 주세요.")
        if not still_image and args.output.suffix.lower() != ".mp4":
            parser.error("웹캠/동영상 출력은 .mp4 확장자를 사용해 주세요.")
        if not camera and args.output.resolve() == source_path.resolve():
            parser.error("출력 경로는 입력 경로와 달라야 합니다.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.json_output:
        protected = [source_path.resolve()] if not camera else []
        protected.extend([args.classifier.resolve(), args.pose_model.resolve()])
        if args.output:
            protected.append(args.output.resolve())
        if args.json_output.resolve() in protected:
            parser.error("JSON 결과 경로는 입력/모델/영상 출력과 달라야 합니다.")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise ValueError("인식 패키지가 없습니다. python -m pip install -r requirements-pose.txt 실행") from exc

    model = YOLO(str(args.pose_model))
    capture, writer, log = None, None, None
    frame_number = 0
    preview_size = None
    preview_window = "Pose Recognition - q / ESC to quit"
    try:
        if args.json_output:
            log = args.json_output.open("w", encoding="utf-8")
        if still_image:
            # Windows 한글 경로를 지원한다.
            image = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {source_path}")
        else:
            if camera:
                # 연결을 확인한 외장 웹캠은 1번이며 Windows에서는 DirectShow를 사용한다.
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                capture = cv2.VideoCapture(int(args.source), backend)
            else:
                capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise ValueError(f"카메라 또는 동영상을 열 수 없습니다: {args.source}")
            if camera:
                print(f"사용 중인 웹캠: {args.source}번 (--source 번호로 변경 가능)", flush=True)
        while True:
            if still_image:
                frame = image
            else:
                success, frame = capture.read()
                if not success:
                    if frame_number == 0:
                        raise ValueError("입력에서 프레임을 읽을 수 없습니다.")
                    break
            frame = apply_digital_zoom(frame, args.zoom)
            result = model.predict(frame, conf=args.detection_confidence, device=args.device, verbose=False)[0]
            records = classify_result(result, classifier, args.threshold, args.keypoint_confidence)
            if log:
                log.write(json.dumps({"frame": frame_number, "people": records}, ensure_ascii=False) + "\n")
                log.flush()
            if still_image:
                print(json.dumps({"source": str(source_path), "people": records}, ensure_ascii=False, indent=2))
            annotated = None
            if args.output:
                annotated = annotate(result, records, args.debug, args.threshold, args.zoom)
                if still_image:
                    success, encoded = cv2.imencode(args.output.suffix, annotated)
                    if not success:
                        raise ValueError("이미지 인코딩에 실패했습니다.")
                    encoded.tofile(args.output)
                else:
                    if writer is None:
                        fps = float(capture.get(cv2.CAP_PROP_FPS))
                        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (annotated.shape[1], annotated.shape[0]),
                        )
                        if not writer.isOpened():
                            raise ValueError(f"출력 동영상을 생성할 수 없습니다: {args.output}")
                    writer.write(annotated)
            frame_number += 1
            if not args.no_display:
                if camera or annotated is None:
                    annotated = annotate(result, records, args.debug, args.threshold, args.zoom, mirror=camera)
                if frame_number == 1:
                    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                    if sys.platform == "win32":
                        # 저장된 창 위치가 다른 모니터여도 기본 모니터에서 전체 화면으로 전환한다.
                        cv2.resizeWindow(preview_window, 320, 240)
                        cv2.moveWindow(preview_window, 0, 0)
                    cv2.setWindowProperty(preview_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    if sys.platform == "win32":
                        _, _, width, height = cv2.getWindowImageRect(preview_window)
                        if width > 0 and height > 0:
                            preview_size = (width, height)
                preview = fit_fullscreen_preview(annotated, *preview_size) if preview_size else annotated
                cv2.imshow(preview_window, preview)
                if cv2.waitKey(0 if still_image else 1) & 0xFF in (ord("q"), 27):
                    break
            if still_image:
                break
        if args.output:
            print(f"저장 완료: {args.output.resolve()}")
        print(f"처리한 프레임: {frame_number}")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if log is not None:
            log.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"인식 실패: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n인식을 종료했습니다.")
