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

from model_utils import prep_inference, run_inference_and_extract_soft_targets, crops_to_tensor

import logging
import warnings

from student_model import StudentLipReader, DistillationTrainer

import torch

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

# Derive dimensions from real teacher output
print(f"\nTeacher soft targets shape: {soft_targets.shape}")
VOCAB_SIZE = soft_targets.size(-1)
SEQ_LEN = soft_targets.size(1)
BATCH_SIZE = 1

# Get special token indices from teacher's dictionary
target_dict = task.target_dictionary
PAD_IDX = target_dict.pad()
BOS_IDX = target_dict.eos()  # fairseq uses EOS token as BOS for decoder input
print(f"Vocab size: {VOCAB_SIZE}, Seq len: {SEQ_LEN}, PAD: {PAD_IDX}, BOS: {BOS_IDX}")

# Derive hard targets from teacher's best predictions
hard_targets = soft_targets.argmax(dim=-1)  # (1, SEQ_LEN)

# Construct prev_tokens: [BOS, tok_0, tok_1, ...] (shifted right for teacher forcing)
prev_tokens = torch.cat([
    torch.full((BATCH_SIZE, 1), BOS_IDX, dtype=torch.long),
    hard_targets[:, :-1],
], dim=1)

# Prepare video frames from real crops
video_frames = crops_to_tensor(crops)
print(f"Video frames shape: {video_frames.shape}")

# Initialize student model with teacher's vocab size
student = StudentLipReader(
    vocab_size=VOCAB_SIZE,
    embed_dim=256,
    encoder_layers=4,
    decoder_layers=4,
    n_heads=4,
    ff_dim=512,
    pad_idx=PAD_IDX,
    freeze_early_resnet=True,
)
print()
student.print_parameter_breakdown()

# Run training step with real teacher soft targets
trainer = DistillationTrainer(student, temperature=2.0, alpha=0.7)
losses = trainer.train_step(video_frames, prev_tokens, soft_targets, hard_targets)

print(f"\nCombined loss: {losses['loss']:.4f}")
print(f"Soft loss:     {losses['soft_loss']:.4f}")
print(f"Hard loss:     {losses['hard_loss']:.4f}")
