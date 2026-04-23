"""Pure-tensor preprocessing tests.

Runs locally with just torch + numpy + cv2 — no fairseq, no AV-HuBERT
model weights. Imports av_hubert/avhubert/utils.py directly by path so
we can assert byte-for-byte equivalence with the real teacher pipeline.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
AV_HUBERT_UTILS_PATH = REPO_ROOT.parent / "av_hubert" / "avhubert" / "utils.py"

sys.path.insert(0, str(REPO_ROOT))
from preprocessing import (  # noqa: E402
    AVHUBERT_CROP_SIZE,
    AVHUBERT_IMAGE_MEAN,
    AVHUBERT_IMAGE_STD,
    _center_crop,
    crops_to_tensor,
)


def _load_avhubert_utils():
    if not AV_HUBERT_UTILS_PATH.exists():
        pytest.skip(f"av_hubert utils.py not found at {AV_HUBERT_UTILS_PATH}")
    spec = importlib.util.spec_from_file_location(
        "avhubert_utils", AV_HUBERT_UTILS_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_output_shape_and_dtype():
    crops = np.full((10, 96, 96), 128, dtype=np.uint8)
    out = crops_to_tensor(crops)
    assert out.shape == (1, 10, 1, 88, 88)
    assert out.dtype == torch.float32


def test_normalization_midgray():
    crops = np.full((1, 96, 96), 128, dtype=np.uint8)
    out = crops_to_tensor(crops)
    expected = (128.0 / 255.0 - AVHUBERT_IMAGE_MEAN) / AVHUBERT_IMAGE_STD
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6)


def test_normalization_edges():
    zeros = crops_to_tensor(np.zeros((1, 96, 96), dtype=np.uint8))
    full = crops_to_tensor(np.full((1, 96, 96), 255, dtype=np.uint8))
    expected_zero = -AVHUBERT_IMAGE_MEAN / AVHUBERT_IMAGE_STD
    expected_full = (1.0 - AVHUBERT_IMAGE_MEAN) / AVHUBERT_IMAGE_STD
    assert torch.allclose(zeros, torch.full_like(zeros, expected_zero), atol=1e-6)
    assert torch.allclose(full, torch.full_like(full, expected_full), atol=1e-6)


def test_center_crop_extracts_inner_88x88():
    # 4-pixel ring of 255s at the boundary, 0s inside. After 88x88
    # center-crop of a 96x96 frame (delta=4), the ring is fully removed.
    frame = np.zeros((2, 96, 96), dtype=np.uint8)
    frame[:, :4, :] = 255
    frame[:, -4:, :] = 255
    frame[:, :, :4] = 255
    frame[:, :, -4:] = 255
    cropped = _center_crop(frame, AVHUBERT_CROP_SIZE)
    assert cropped.shape == (2, 88, 88)
    assert (cropped == 0).all()


def test_matches_avhubert_compose():
    """Gold-standard: crops_to_tensor must equal AV-HuBERT's own Compose."""
    av_utils = _load_avhubert_utils()
    transform = av_utils.Compose([
        av_utils.Normalize(0.0, 255.0),
        av_utils.CenterCrop((AVHUBERT_CROP_SIZE, AVHUBERT_CROP_SIZE)),
        av_utils.Normalize(AVHUBERT_IMAGE_MEAN, AVHUBERT_IMAGE_STD),
    ])

    rng = np.random.default_rng(42)
    crops = rng.integers(0, 256, size=(7, 96, 96), dtype=np.uint8)

    ours = crops_to_tensor(crops)
    reference = torch.from_numpy(transform(crops).astype(np.float32)).unsqueeze(0).unsqueeze(2)

    assert ours.shape == reference.shape
    assert torch.allclose(ours, reference, atol=1e-6)
