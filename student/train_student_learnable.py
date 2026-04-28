"""C4 — learnable viseme-conditioned temperatures.

Identical to Sebastian's vctemp train_student.py except: instead of loading
fixed per-token temperatures from disk, we expose 14 learnable log-temperatures
(one per viseme category) and build a (vocab_size,) lookup table that maps each
token to its viseme bucket via g2p_en + Sebastian's exact PHONEME_TO_VISEME map.

Each forward pass computes token_temperatures = exp(log_temps)[token_to_viseme]
on the fly, so autograd flows through into log_temps. After training, log_temps
encodes the model's learned answer to "should bilabials get higher or lower T
than labiodentals?" — exactly the Hyp A vs Hyp B question, answered by gradient
descent rather than by hand-coding a temperature table.

Initialised so all 14 visemes share T = args.temperature (uniform), so step 0
matches Sebastian's uniform-T baseline exactly.
"""
import argparse
import math
import os
import time
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

AVHUBERT_PARENT = (Path(__file__).parent / "../../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq  # type: ignore
from avhubert import hubert_pretraining  # type: ignore  # noqa
from avhubert import hubert  # type: ignore  # noqa
from avhubert import hubert_asr  # type: ignore  # noqa

from student_dataset import (
    LRWDistillationDataset, collate_fn, split_clip_ids_by_lrw_split,
)
from student_model import VideoStudent


# Viseme bucketing matches Sebastian's analysis notebook (and the report).
VISEME_LIST = [
    'bilabial', 'labiodental', 'dental', 'alveolar', 'postalveolar',
    'velar', 'glottal', 'approximant', 'front_unrounded', 'open',
    'rounded', 'diphthong', 'rhotic_vowel', 'unknown',
]
N_VISEMES = len(VISEME_LIST)
VISEME_TO_IDX = {v: i for i, v in enumerate(VISEME_LIST)}

PHONEME_TO_VISEME = {
    'B': 'bilabial', 'P': 'bilabial', 'M': 'bilabial',
    'F': 'labiodental', 'V': 'labiodental',
    'TH': 'dental', 'DH': 'dental',
    'T': 'alveolar', 'D': 'alveolar', 'N': 'alveolar',
    'S': 'alveolar', 'Z': 'alveolar', 'L': 'alveolar',
    'SH': 'postalveolar', 'ZH': 'postalveolar',
    'CH': 'postalveolar', 'JH': 'postalveolar',
    'K': 'velar', 'G': 'velar', 'NG': 'velar',
    'HH': 'glottal',
    'R': 'approximant', 'W': 'approximant', 'Y': 'approximant',
    'IY': 'front_unrounded', 'IH': 'front_unrounded', 'EH': 'front_unrounded',
    'EY': 'front_unrounded', 'AE': 'front_unrounded',
    'AA': 'open', 'AH': 'open', 'AO': 'open',
    'OW': 'rounded', 'UH': 'rounded', 'UW': 'rounded',
    'AW': 'diphthong', 'AY': 'diphthong', 'OY': 'diphthong',
    'ER': 'rhotic_vowel',
}


def build_token_viseme_table(dictionary, device):
    """Map each vocab token to a viseme index based on its first phoneme.
    Tokens that aren't real words (special tokens, punctuation, single-letter
    subwords without phonemes) get the 'unknown' bucket."""
    from g2p_en import G2p
    g2p = G2p()
    vocab_size = len(dictionary)
    table = torch.full((vocab_size,), VISEME_TO_IDX['unknown'],
                       dtype=torch.long, device=device)
    for token_id in range(vocab_size):
        token_str = dictionary.string([token_id]).replace('▁', '').strip()
        if not token_str:
            continue
        if token_str.startswith('<') or token_str.endswith('>'):
            continue
        if not any(c.isalpha() for c in token_str):
            continue
        try:
            phones = g2p(token_str)
        except Exception:
            continue
        # First phonetic phoneme (skip non-phonemes returned by g2p like
        # punctuation or word-boundary markers).
        first = None
        for p in phones:
            stripped = ''.join(c for c in p if not c.isdigit())
            if stripped in PHONEME_TO_VISEME:
                first = stripped
                break
        if first is None:
            continue
        table[token_id] = VISEME_TO_IDX[PHONEME_TO_VISEME[first]]
    return table


def get_special_token_ids(ckpt_path):
    _, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    d = task.target_dictionary
    return {
        'vocab_size': len(d),
        'pad_id': d.pad(),
        'bos_id': d.bos(),
        'eos_id': d.eos(),
    }, d


def kd_loss(student_logits, teacher_logits, teacher_tokens, decoder_mask,
            temperature=2.0, alpha=0.5, pad_id=1, token_temperatures=None):
    """Same as Sebastian's kd_loss. token_temperatures may be a tensor with
    gradient — autograd flows back to log_temps."""
    if token_temperatures is not None:
        T = token_temperatures[teacher_tokens].unsqueeze(-1)  # (B, T_d, 1)
    else:
        T = temperature

    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_soft = F.softmax(teacher_logits / T, dim=-1)
    kl = -(t_soft * s_log).sum(dim=-1)  # (B, T_d)

    if isinstance(T, torch.Tensor):
        kl = kl * (T.squeeze(-1) ** 2)
    else:
        kl = kl * (T ** 2)

    valid = (~decoder_mask).float()
    kd_term = (kl * valid).sum() / valid.sum().clamp(min=1.0)

    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_tokens.reshape(-1),
        ignore_index=pad_id,
        reduction='mean',
    )
    return alpha * kd_term + (1 - alpha) * ce, kd_term.item(), ce.item()


def shift_right_for_teacher_forcing(tokens, bos_id, pad_id):
    B, T = tokens.shape
    bos = torch.full((B, 1), bos_id, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([bos, tokens[:, :-1]], dim=1)


@torch.no_grad()
def eval_kl(model, loader, special, device, temperature=1.0):
    model.eval()
    total_kl, total_tokens = 0.0, 0
    for batch in loader:
        video = batch['video'].to(device)
        video_mask = batch['video_mask'].to(device)
        teacher_logits = batch['teacher_logits'].to(device)
        teacher_tokens = batch['teacher_tokens'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        prev_tokens = shift_right_for_teacher_forcing(
            teacher_tokens, special['bos_id'], special['pad_id'])
        student_logits = model(video, video_mask, prev_tokens,
                               decoder_mask=decoder_mask)
        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        kl_per_tok = -(t_soft * s_log).sum(dim=-1)
        valid = (~decoder_mask).float()
        total_kl += (kl_per_tok * valid).sum().item()
        total_tokens += valid.sum().item()
    return total_kl / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--ckpt', default='data/checkpoints/checkpoint.pt')
    ap.add_argument('--out_dir', default='runs/c4_learnable')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--temperature', type=float, default=2.0,
                    help='Initial T for all 14 viseme buckets. log_temps starts '
                         'at log(temperature) so step 0 == uniform-T baseline.')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--enc_layers', type=int, default=4)
    ap.add_argument('--dec_layers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading teacher dictionary for vocab info...")
    special, d = get_special_token_ids(args.ckpt)
    print(f"  vocab_size={special['vocab_size']} "
          f"pad={special['pad_id']} bos={special['bos_id']} eos={special['eos_id']}")

    print("Building token -> viseme map (one-time, ~1 min)...")
    token_to_viseme = build_token_viseme_table(d, device)
    counts = Counter(token_to_viseme.cpu().tolist())
    print(f"  vocab tokens by viseme bucket:")
    for i, name in enumerate(VISEME_LIST):
        print(f"    {i:2d} {name:<16} {counts.get(i, 0)}")

    log_temps = nn.Parameter(torch.full(
        (N_VISEMES,), math.log(args.temperature),
        device=device, dtype=torch.float32,
    ))
    print(f"  log_temps initialised so all 14 visemes start at "
          f"T={args.temperature:.2f} (matches uniform-T baseline at step 0).")

    print("Setting up datasets...")
    full = LRWDistillationDataset(args.videos, args.logits)
    splits = split_clip_ids_by_lrw_split(full.clip_ids)
    train_set = LRWDistillationDataset(args.videos, args.logits,
                                        clip_ids=splits['train'])
    val_set = LRWDistillationDataset(args.videos, args.logits,
                                      clip_ids=splits['val'])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    model = VideoStudent(
        vocab_size=special['vocab_size'],
        dim=args.dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        pad_id=special['pad_id'],
        bos_id=special['bos_id'],
        eos_id=special['eos_id'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Student: {n_params:.1f}M params + 14 learnable log-temperatures")

    trainable = list(model.parameters()) + [log_temps]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_kl = float('inf')
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = running_kd = running_ce = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            video = batch['video'].to(device, non_blocking=True)
            video_mask = batch['video_mask'].to(device, non_blocking=True)
            teacher_logits = batch['teacher_logits'].to(device, non_blocking=True)
            teacher_tokens = batch['teacher_tokens'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)

            prev_tokens = shift_right_for_teacher_forcing(
                teacher_tokens, special['bos_id'], special['pad_id'])

            # Compute the (V,) per-token temperature tensor from learnable
            # log_temps. Gradient flows back into log_temps via the indexing.
            token_temperatures = torch.exp(log_temps)[token_to_viseme]  # (V,)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                student_logits = model(video, video_mask, prev_tokens,
                                       decoder_mask=decoder_mask)
                loss, kd_val, ce_val = kd_loss(
                    student_logits.float(), teacher_logits, teacher_tokens,
                    decoder_mask,
                    temperature=args.temperature, alpha=args.alpha,
                    pad_id=special['pad_id'],
                    token_temperatures=token_temperatures,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            running_kd += kd_val
            running_ce += ce_val
            n_batches += 1
            pbar.set_postfix({
                'loss': f"{running_loss/n_batches:.3f}",
                'kd': f"{running_kd/n_batches:.3f}",
                'ce': f"{running_ce/n_batches:.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })

        train_loss = running_loss / n_batches
        elapsed = time.time() - t0

        val_kl = eval_kl(model, val_loader, special, device,
                         temperature=args.temperature)

        # Print learned per-viseme temperatures (the report's headline result).
        with torch.no_grad():
            temps = torch.exp(log_temps).detach().cpu().tolist()
        print(f"epoch {epoch+1}: train_loss={train_loss:.3f} "
              f"val_kl(T={args.temperature})={val_kl:.3f} time={elapsed:.0f}s")
        print("  learned T per viseme:")
        for i, name in enumerate(VISEME_LIST):
            print(f"    T[{name:<14}] = {temps[i]:.3f}")

        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_kl': val_kl,
            'args': vars(args),
            'log_temps': log_temps.detach().cpu(),
            'viseme_list': VISEME_LIST,
            'token_to_viseme': token_to_viseme.cpu(),
        }
        torch.save(ckpt, os.path.join(args.out_dir, 'last.pt'))
        if val_kl < best_val_kl:
            best_val_kl = val_kl
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            print(f"  → new best val_kl={val_kl:.3f}, saved best.pt")


if __name__ == '__main__':
    main()
