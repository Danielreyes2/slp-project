import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from python_speech_features import logfbank

# Path setup for AV-HuBERT
AVHUBERT_PARENT = (Path(__file__).parent / "../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq  # type: ignore
from avhubert import hubert_pretraining  # type: ignore  # noqa
from avhubert import hubert  # type: ignore  # noqa
from avhubert import hubert_asr  # type: ignore  # noqa


# ----- Audio preprocessing (replicates AV-HuBERT's hubert_dataset transforms) -----

def stacker(feats, stack_order):
    """Concatenate consecutive feature frames. Mirrors AV-HuBERT exactly."""
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res_zeros = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res_zeros], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats


def extract_audio_features(wav_data, video_len, sample_rate=16000,
                           stack_order=4, normalize=True):
    """logfbank → stacker → length-match to video → layer_norm."""
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # (T, 26)
    audio_feats = stacker(audio_feats, stack_order)  # (T/4, 104)

    # Length-match to video
    diff = len(audio_feats) - video_len
    if diff < 0:
        audio_feats = np.concatenate([
            audio_feats,
            np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)
        ])
    elif diff > 0:
        audio_feats = audio_feats[:-diff]

    audio_t = torch.from_numpy(audio_feats)
    if normalize:
        with torch.no_grad():
            audio_t = F.layer_norm(audio_t, audio_t.shape[1:])
    return audio_t  # (video_len, 104)


# ----- Video preprocessing -----

def normalize_video(video):
    """Grayscale (T, 96, 96) uint8 → (T, 1, 96, 96) float, normalized."""
    x = torch.from_numpy(video).float() / 255.0
    mean, std = 0.421, 0.165  # AV-HuBERT video stats
    x = (x - mean) / std
    return x.unsqueeze(1)


# ----- Data loading & batching -----

def find_npz_files(roi_dir):
    paths = []
    for root, _, files in os.walk(roi_dir):
        for f in files:
            if f.endswith('.npz'):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def get_clip_id(npz_path, roi_dir):
    rel = os.path.relpath(npz_path, roi_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    return rel_no_ext.replace('/', '_')


def load_clip(npz_path, roi_dir):
    data = np.load(npz_path)
    video_np = data['video']  # (T, 96, 96)
    video_len = video_np.shape[0]

    video = normalize_video(video_np)  # (T, 1, 96, 96)
    audio_raw = data['audio'].astype(np.float32)
    audio = extract_audio_features(audio_raw, video_len)  # (T, 104)

    cid = get_clip_id(npz_path, roi_dir)
    return video, audio, cid


def collate_batch(items):
    """Pad variable-length clips. Audio and video share T after stacker matching."""
    videos, audios, ids = zip(*items)
    v_lens = [v.shape[0] for v in videos]
    max_v = max(v_lens)
    feat_dim = audios[0].shape[-1]  # 104

    B = len(videos)
    video_batch = torch.zeros(B, max_v, 1, 96, 96)
    audio_batch = torch.zeros(B, max_v, feat_dim)
    pad_mask = torch.ones(B, max_v, dtype=torch.bool)

    for i, (v, a) in enumerate(zip(videos, audios)):
        T = v.shape[0]
        video_batch[i, :T] = v
        audio_batch[i, :T] = a
        pad_mask[i, :T] = False

    return video_batch, audio_batch, pad_mask, list(ids), v_lens


# ----- Inference: run encoder, then autoregressive greedy decoder loop -----

@torch.no_grad()
def get_decoder_logits(model, video_batch, audio_batch, pad_mask,
                      max_decode_len=50):
    """
    Returns:
        logits: (B, T_dec, vocab) raw decoder logits, one row per generated token
        tokens: (B, T_dec) greedy-chosen tokens
        lengths: list of valid T_dec per sample (up to and including EOS)
    """
    device = next(model.parameters()).device

    video = video_batch.permute(0, 2, 1, 3, 4).to(device).half()  # (B, 1, T, 96, 96)
    audio = audio_batch.permute(0, 2, 1).to(device).half()        # (B, 104, T)
    pad_mask_dev = pad_mask.to(device)

    source = {'video': video, 'audio': audio}

    encoder_out = model.encoder(source=source, padding_mask=pad_mask_dev)

    # Decoder dictionary lookup
    tgt_dict = getattr(model.decoder, 'dictionary', None)
    if tgt_dict is None:
        raise RuntimeError("Decoder has no dictionary attribute; check model API.")

    bos = tgt_dict.bos()
    eos = tgt_dict.eos()

    B = video.shape[0]
    prev_tokens = torch.full((B, 1), bos, dtype=torch.long, device=device)
    all_logits = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = [0] * B

    for step in range(max_decode_len):
        decoder_out = model.decoder(
            prev_output_tokens=prev_tokens,
            encoder_out=encoder_out,
        )
        logits = decoder_out[0] if isinstance(decoder_out, tuple) else decoder_out
        step_logits = logits[:, -1, :]  # (B, vocab) — only the new token's distribution
        all_logits.append(step_logits.float().cpu())

        next_tokens = step_logits.argmax(dim=-1)
        prev_tokens = torch.cat([prev_tokens, next_tokens.unsqueeze(1)], dim=1)

        for i in range(B):
            if not finished[i]:
                lengths[i] = step + 1
                if next_tokens[i].item() == eos:
                    finished[i] = True

        if finished.all():
            break

    logits_tensor = torch.stack(all_logits, dim=1)  # (B, T_dec, vocab)
    return logits_tensor, prev_tokens[:, 1:].cpu(), lengths


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output', default='logits.h5')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only first N clips (for testing)')
    ap.add_argument('--max_decode_len', type=int, default=50)
    args = ap.parse_args()

    print(f"Loading model from {args.ckpt}")
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt])
    model = models[0]
    model.eval().cuda().half()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    if hasattr(task, 'target_dictionary'):
        print(f"Vocab size: {len(task.target_dictionary)}")

    paths = find_npz_files(args.roi_dir)
    if args.limit:
        paths = paths[:args.limit]
    print(f"Processing {len(paths)} clips")

    with h5py.File(args.output, 'w') as h5f:
        for i in tqdm(range(0, len(paths), args.batch_size)):
            batch_paths = paths[i:i + args.batch_size]

            items = []
            for p in batch_paths:
                try:
                    items.append(load_clip(p, args.roi_dir))
                except Exception as e:
                    print(f"  load failed: {p}: {e}")

            if not items:
                continue

            video_batch, audio_batch, pad_mask, ids, _ = collate_batch(items)

            try:
                logits, tokens, lengths = get_decoder_logits(
                    model, video_batch, audio_batch, pad_mask,
                    max_decode_len=args.max_decode_len,
                )
            except Exception as e:
                print(f"  inference failed for batch starting at {batch_paths[0]}: {e}")
                import traceback
                traceback.print_exc()
                continue

            for j, cid in enumerate(ids):
                valid_len = lengths[j]
                logits_j = logits[j, :valid_len].numpy().astype(np.float16)  # (T_dec, vocab)
                tokens_j = tokens[j, :valid_len].numpy()

                grp = h5f.create_group(cid)
                grp.create_dataset('logits', data=logits_j,
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('tokens', data=tokens_j)

    print(f"\nDone. Saved to {args.output}")
    with h5py.File(args.output, 'r') as h5f:
        keys = list(h5f.keys())
        print(f"Total clips saved: {len(keys)}")
        if keys:
            sample_key = keys[0]
            sample = h5f[sample_key]
            print(f"Example clip {sample_key}:")
            print(f"  logits: {sample['logits'].shape} {sample['logits'].dtype}")
            print(f"  tokens: {sample['tokens'].shape}")


if __name__ == '__main__':
    main()
