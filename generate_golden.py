"""Generate the cached teacher-logit fixture for the preprocessing lock test.

Runs once (typically on Colab) against AFTERNOON.mp4 + the AVSR-finetuned
AV-HuBERT checkpoint. Saves soft_targets + metadata to
tests/fixtures/afternoon_teacher_logits.pt so the assertion test can
verify determinism of the full pipeline on later runs.

Re-run whenever preprocessing intentionally changes. Commit the new
fixture in the same PR as the preprocessing change.
"""
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
FAIRSEQ_PATH = REPO_ROOT / "../av_hubert/fairseq"
AV_HUBERT_PATH = REPO_ROOT / "../av_hubert/avhubert"
sys.path.insert(0, str(FAIRSEQ_PATH))
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq  # type: ignore  # noqa: E402

from mp import process_video, save_crops_as_video  # noqa: E402
from model_utils import (  # noqa: E402
    prep_inference,
    run_inference_and_extract_soft_targets,
)

CKPT = "checkpoint.pt"
VIDEO = "AFTERNOON.mp4"
ROI_VIDEO = "AFTERNOON-roi.mp4"
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/afternoon_teacher_logits.pt"


def main():
    assert Path(CKPT).exists(), f"Missing {CKPT} — download per README."
    assert Path(VIDEO).exists(), f"Missing {VIDEO} — download per README."

    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([CKPT])
    model = models[0].eval()

    crops, fps = process_video(VIDEO)
    save_crops_as_video(crops, ROI_VIDEO)

    itr, _, _ = prep_inference(os.path.abspath(ROI_VIDEO), model, cfg, task)
    soft_targets = run_inference_and_extract_soft_targets(model, itr)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "soft_targets": soft_targets.cpu(),
            "shape": tuple(soft_targets.shape),
            "num_crops": int(crops.shape[0]),
            "fps": float(fps),
            "video": VIDEO,
            "checkpoint": CKPT,
            "device": str(soft_targets.device),
            "torch_version": torch.__version__,
        },
        FIXTURE_PATH,
    )
    print(f"Saved fixture -> {FIXTURE_PATH}")
    print(f"  shape={soft_targets.shape} device={soft_targets.device}")


if __name__ == "__main__":
    main()
