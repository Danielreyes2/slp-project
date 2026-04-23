"""Full-pipeline teacher-logit determinism assertion.

Colab-only: needs fairseq, AV-HuBERT checkpoint, and AFTERNOON.mp4.
Re-runs MediaPipe -> CenterCrop -> AV-HuBERT on the same input and
asserts the result matches the cached golden tensor within float32
tolerance. If this fails, preprocessing has drifted — fix before doing
anything else downstream.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/afternoon_teacher_logits.pt"

FAIRSEQ_PATH = REPO_ROOT / "../av_hubert/fairseq"
AV_HUBERT_PATH = REPO_ROOT / "../av_hubert/avhubert"


@pytest.fixture(scope="module")
def teacher_pipeline():
    try:
        sys.path.insert(0, str(FAIRSEQ_PATH))
        sys.path.insert(0, str(AV_HUBERT_PATH))
        sys.path.insert(0, str(REPO_ROOT))
        import fairseq  # type: ignore
        from mp import process_video, save_crops_as_video
        from model_utils import prep_inference, run_inference_and_extract_soft_targets
    except Exception as e:
        pytest.skip(f"fairseq / av_hubert not importable: {e}")

    return {
        "fairseq": fairseq,
        "process_video": process_video,
        "save_crops_as_video": save_crops_as_video,
        "prep_inference": prep_inference,
        "run_inference_and_extract_soft_targets": run_inference_and_extract_soft_targets,
    }


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="fixture missing — run `python generate_golden.py` first",
)
def test_teacher_logits_deterministic(teacher_pipeline):
    cached = torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)
    cached_targets = cached["soft_targets"]

    ckpt = str(REPO_ROOT / "checkpoint.pt")
    video = str(REPO_ROOT / "AFTERNOON.mp4")
    roi = str(REPO_ROOT / "AFTERNOON-roi.mp4")

    assert Path(ckpt).exists(), f"missing {ckpt}"
    assert Path(video).exists(), f"missing {video}"

    models, cfg, task = teacher_pipeline["fairseq"].checkpoint_utils.load_model_ensemble_and_task(
        [ckpt]
    )
    model = models[0].eval()

    crops, _ = teacher_pipeline["process_video"](video)
    teacher_pipeline["save_crops_as_video"](crops, roi)

    itr, _, _ = teacher_pipeline["prep_inference"](os.path.abspath(roi), model, cfg, task)
    new_targets = teacher_pipeline["run_inference_and_extract_soft_targets"](model, itr).cpu()

    assert new_targets.shape == cached_targets.shape, (
        f"shape drift: new={new_targets.shape} cached={cached_targets.shape}"
    )
    max_diff = (new_targets - cached_targets).abs().max().item()
    assert torch.allclose(new_targets, cached_targets, atol=1e-4), (
        f"teacher logits drifted: max abs diff = {max_diff:.2e} (atol=1e-4). "
        "Preprocessing changed — fix before continuing."
    )
