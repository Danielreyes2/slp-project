"""AV-HuBERT inference preprocessing, isolated from fairseq.

Kept separate from model_utils.py so unit tests can import it without
pulling in fairseq / hubert_pretraining / etc. — those are heavy and
Python-version-sensitive, but the pixel pipeline is pure numpy/torch.
"""
import numpy as np
import torch

# Must match avhubert/hubert_pretraining.py (image_mean=0.421,
# image_std=0.165, image_crop_size=88) and the inference Compose in
# avhubert/hubert_dataset.py:223-226:
#   Normalize(0, 255) -> CenterCrop(88, 88) -> Normalize(mean, std)
# Drift here silently poisons every downstream KD signal.
AVHUBERT_IMAGE_MEAN = 0.421
AVHUBERT_IMAGE_STD = 0.165
AVHUBERT_CROP_SIZE = 88


def _center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    # Mirrors avhubert/utils.py::CenterCrop exactly. The
    # int(round((h-th))/2.) rounding is load-bearing — any other rounding
    # shifts pixels by one and breaks the teacher-logit assertion.
    t, h, w = frames.shape
    delta_h = int(round((h - size)) / 2.)
    delta_w = int(round((w - size)) / 2.)
    return frames[:, delta_h:delta_h + size, delta_w:delta_w + size]


def crops_to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(T, H, W) uint8 mouth crops -> (1, T, 1, 88, 88) float32 tensor.

    Replicates AV-HuBERT's inference Compose exactly so the student and
    teacher see the same pixels.
    """
    frames = crops.astype(np.float32) / 255.0
    frames = _center_crop(frames, AVHUBERT_CROP_SIZE)
    frames = (frames - AVHUBERT_IMAGE_MEAN) / AVHUBERT_IMAGE_STD
    return torch.FloatTensor(frames).unsqueeze(0).unsqueeze(2)
