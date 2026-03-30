import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

def extract_lip_crop(frame, detector, target_size=(96, 96)):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    h, w = frame.shape[:2]
    LIP_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
    ]
    lip_points = []
    for idx in LIP_INDICES:
        lm = landmarks[idx]
        lip_points.append((int(lm.x * w), int(lm.y * h)))
    lip_points = np.array(lip_points)
    x_min, y_min = lip_points.min(axis=0)
    x_max, y_max = lip_points.max(axis=0)
    pad_x = int((x_max - x_min) * 0.3)
    pad_y = int((y_max - y_min) * 0.3)
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)
    crop = frame[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized

def process_video(video_path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}. Check the file exists and was uploaded.")
        return np.array([]), 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or frame_count == 0:
        print(f"ERROR: Video has 0 fps or 0 frames. File may be corrupted or missing.")
        cap.release()
        return np.array([]), 0

    print(f"Video: {fps:.1f} fps, {frame_count} frames")
    crops = []
    failed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = extract_lip_crop(rgb, detector)
        if crop is not None:
            crops.append(crop)
        else:
            failed += 1
    cap.release()
    detector.close()
    total = len(crops) + failed
    if total > 0:
        print(f"Extracted {len(crops)}/{total} frames ({failed} failed, {failed/total*100:.1f}% drop rate)")
        if failed / total > 0.20:
            print("WARNING: >20% drop rate — excluded per project spec")
        if abs(fps - 25) > 1:
            print(f"WARNING: {fps:.0f}fps recorded, model expects 25fps — resampling needed")
    return np.array(crops) if crops else np.array([]), fps

def resample_to_25fps(crops, source_fps=30, target_fps=25):
    if source_fps == 0 or len(crops) == 0:
        print("ERROR: No crops or 0 fps — cannot resample.")
        return crops
    n_source = len(crops)
    duration = n_source / source_fps
    n_target = int(duration * target_fps)
    source_indices = np.linspace(0, n_source - 1, n_target)
    resampled = np.array([crops[int(round(i))] for i in source_indices])
    print(f"Resampled: {n_source} frames @ {source_fps:.0f}fps -> {len(resampled)} frames @ {target_fps}fps")
    return resampled

def save_crops_as_video(crops, output_path, fps=25):
    h, w = crops[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    for frame in crops:
        out.write(frame)
    out.release()
    print(f"Saved {len(crops)} frames to {output_path}")
