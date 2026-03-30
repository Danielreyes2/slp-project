import sys
from pathlib import Path

FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(FAIRSEQ_PATH))

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq # type: ignore
import hubert_pretraining, hubert, hubert_asr # type: ignore

ckpt_path = "checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval()

import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
if not os.path.exists("face_landmarker.task"):
    urllib.request.urlretrieve(url, "face_landmarker.task")

from mp import process_video, save_crops_as_video

video_path = "AFTERNOON.mp4"
crops, fps = process_video(video_path)

import matplotlib.pyplot as plt

if len(crops) > 0:
    num_show = min(8, len(crops))
    fig, axes = plt.subplots(1, num_show, figsize=(16, 3))
    for i, ax in enumerate(axes):
        idx = i * (len(crops) // num_show)
        ax.imshow(crops[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f"F{idx}")
    plt.suptitle(f"Lip crops (resampled to 25fps, {len(crops)} frames)")
    plt.savefig("crops.png")

save_crops_as_video(crops, "AFTERNOON-roi.mp4")

from model_utils import predict

hypo = predict(os.path.abspath("AFTERNOON-roi.mp4"), model, cfg, task)
print(hypo)
