import sys
import os
import argparse
from pathlib import Path
import tempfile
import numpy as np
import torch
import cv2

AVHUBERT_PARENT = (Path(__file__).parent / "../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq  # type: ignore

from fairseq import tasks, utils  # type: ignore
from fairseq.dataclass.configs import GenerationConfig  # type: ignore
from omegaconf import OmegaConf  # type: ignore

from avhubert import hubert_pretraining  # type: ignore  # noqa
from avhubert import hubert  # type: ignore  # noqa
from avhubert import hubert_asr  # type: ignore  # noqa

def npz_to_temp_video(npz_path):
    """
    Reconstruct an mp4 (with audio) from an .npz so that AV-HuBERT's
    standard dataset/generator pipeline can ingest it.
    Returns (video_mp4_path, audio_wav_path, num_frames).
    """
    import soundfile as sf
    data = np.load(npz_path)
    video = data['video']  # (T, 96, 96) uint8
    audio = data['audio']  # (samples,) float32

    tmp_dir = tempfile.mkdtemp()
    mp4_path = os.path.join(tmp_dir, 'clip.mp4')
    wav_path = os.path.join(tmp_dir, 'clip.wav')

    h, w = video.shape[1], video.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, 25, (w, h), isColor=False)
    for f in video:
        writer.write(f)
    writer.release()

    sf.write(wav_path, audio, 16000)
    return mp4_path, wav_path, len(video)


def setup_inference(ckpt_path, mp4_path, wav_path, num_frames):
    """Mirror the demo's prep_inference but for an .npz-derived clip."""
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval().cuda()

    data_dir = tempfile.mkdtemp()
    audio_samples = int(16000 * num_frames / 25)
    tsv_lines = [
        "/\n",
        f"clip-0\t{mp4_path}\t{wav_path}\t{num_frames}\t{audio_samples}\n",
    ]
    with open(f"{data_dir}/test.tsv", "w") as f:
        f.write("".join(tsv_lines))
    with open(f"{data_dir}/test.wrd", "w") as f:
        f.write("DUMMY\n")

    cfg_dict = OmegaConf.to_container(saved_cfg, resolve=True)
    cfg_dict['task']['modalities'] = ['video', 'audio']
    cfg_dict['task']['data'] = data_dir
    cfg_dict['task']['label_dir'] = data_dir
    cfg_dict['task']['noise_prob'] = 0.0
    cfg_dict['task']['noise_wav'] = None
    cfg = OmegaConf.create(cfg_dict)

    task = tasks.setup_task(cfg.task)
    task.load_dataset('test', task_cfg=cfg.task)

    gen_cfg = GenerationConfig(beam=20)
    generator = task.build_generator([model], gen_cfg)

    def decode_tokens(toks):
        d = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(d.pad())
        return task.datasets['test'].label_processors[0].decode(toks, symbols_ignore)

    itr = task.get_batch_iterator(dataset=task.dataset('test')).next_epoch_itr(shuffle=False)
    return model, task, generator, decode_tokens, itr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='Path to a single .npz clip')
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    mp4_path, wav_path, num_frames = npz_to_temp_video(args.npz)
    model, task, generator, decode_tokens, itr = setup_inference(
        args.ckpt, mp4_path, wav_path, num_frames
    )

    sample = next(iter(itr))
    sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, [model], sample)

    print(f"\nClip: {args.npz}")
    for rank, h in enumerate(hypos[0][:3]):  # top 3 hypotheses
        toks = h['tokens'].int().cpu()
        score = h['score']
        text = decode_tokens(toks)
        print(f"  [{rank}] (score={score:.2f}): {text}")


if __name__ == '__main__':
    main()
