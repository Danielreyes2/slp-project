import sys
from pathlib import Path

AVHUBERT_DIR = (Path(__file__).parent / "../av_hubert/avhubert").resolve()
AVHUBERT_PARENT = (Path(__file__).parent / "../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_DIR))
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import h5py
import fairseq
import hubert_pretraining  # noqa  — top-level, not from avhubert
import hubert  # noqa
import hubert_asr  # noqa


def main():
    ckpt_path = 'data/checkpoints/checkpoint.pt'
    h5_path = 'logits_test.h5'

    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    d = task.target_dictionary

    with h5py.File(h5_path, 'r') as f:
        for cid in list(f.keys())[:15]:
            toks = f[cid]['tokens'][:].tolist()
            text = d.string(toks)
            print(f"{cid}: '{text}'")


if __name__ == '__main__':
    main()
