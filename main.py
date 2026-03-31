import sys
from pathlib import Path

FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(FAIRSEQ_PATH))

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq # type: ignore
import hubert_pretraining, hubert, hubert_asr # type: ignore

import urllib.request
import os

from mp import process_video, save_crops_as_video, plot_crops

from model_utils import predict

# Load AV Hubert model
ckpt_path = "checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval()

# Download mediapipe model
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
if not os.path.exists("face_landmarker.task"):
    urllib.request.urlretrieve(url, "face_landmarker.task")

# Process video using mediapipe
video_path = "AFTERNOON.mp4"
crops, fps = process_video(video_path)

plot_crops(crops)

save_crops_as_video(crops, "AFTERNOON-roi.mp4")

# Run inference
hypo = predict(os.path.abspath("AFTERNOON-roi.mp4"), model, cfg, task)
print(hypo)
