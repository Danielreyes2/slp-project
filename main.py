import sys
from pathlib import Path

FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(FAIRSEQ_PATH))

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq # type: ignore

import urllib.request
import os

from mp import process_video, save_crops_as_video, plot_crops

from model_utils import prep_inference, run_inference_and_extract_soft_targets

import logging
import warnings

# Remove noisy warnings
warnings.filterwarnings("ignore")

# Remove noisy logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("hubert_pretraining").setLevel(logging.WARNING)
logging.getLogger("hubert").setLevel(logging.WARNING)
logging.getLogger("hubert_dataset").setLevel(logging.WARNING)

# Load AV Hubert model
ckpt_path = "checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval()

# print(model)

# Download mediapipe model
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
if not os.path.exists("face_landmarker.task"):
    urllib.request.urlretrieve(url, "face_landmarker.task")

# Process video using mediapipe
video_path = "AFTERNOON.mp4"
crops, fps = process_video(video_path)

plot_crops(crops)

save_crops_as_video(crops, "AFTERNOON-roi.mp4")

# Prepare inference
itr, generator, hypo_token_decoder  = prep_inference(os.path.abspath("AFTERNOON-roi.mp4"), model, cfg, task)

# Run inference and extract soft targets
soft_targets = run_inference_and_extract_soft_targets(model, itr)
print(soft_targets)
print(soft_targets.shape)
