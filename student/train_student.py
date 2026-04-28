import argparse
import math
import os
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

AVHUBERT_PARENT = (Path(__file__).parent / "../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../av_hubert/fairseq").resolve()
# DO NOT add /av_hubert/avhubert to sys.path. The avhubert sources have
# try/except blocks (bare-then-relative imports). With the inner dir on path,
# bare imports succeed and the modules load as TOP-LEVEL, while `from avhubert
# import ...` ALSO loads them as `avhubert.X` — fairseq's `register_model` then
# trips with `Cannot register duplicate model (av_hubert)`. With only
# AVHUBERT_PARENT on path, the bare imports fail, fall to relative imports,
# and the modules load only via the package (single registration).
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


def get_special_token_ids(ckpt_path):
    """Use the teacher's dictionary to get matching BOS/EOS/PAD ids."""
    _, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    d = task.target_dictionary
    return {
        'vocab_size': len(d),
        'pad_id': d.pad(),
        'bos_id': d.bos(),
        'eos_id': d.eos(),
    }


def kd_loss(student_logits, teacher_logits, teacher_tokens, decoder_mask,
            temperature, alpha=0.5, pad_id=1):
    """Combined distillation loss.

    `temperature` is either:
      - a Python float (uniform across all clips), or
      - a tensor of shape (B,) with a per-clip temperature (the learnable
        viseme-conditioned setting). Each clip's temperature applies to every
        decoder position in that clip; the per-clip T² scaling is averaged in
        with the same masking as the rest of the loss.
    """
    # student_logits, teacher_logits: (B, T_d, V)
    # teacher_tokens: (B, T_d)
    # decoder_mask: (B, T_d), True = padded

    if isinstance(temperature, torch.Tensor):
        # Broadcast to (B, 1, 1) for division against (B, T_d, V).
        T_div = temperature.view(-1, 1, 1)
        T_sq = (temperature ** 2).view(-1, 1)  # (B, 1) for per-position scaling
    else:
        T_div = temperature
        T_sq = float(temperature) ** 2  # plain scalar

    # Soft target KD loss (true KL: Σ t · (log t − log s)).
    s_log = F.log_softmax(student_logits / T_div, dim=-1)
    t_soft = F.softmax(teacher_logits / T_div, dim=-1)
    t_log = F.log_softmax(teacher_logits / T_div, dim=-1)
    kl_per_pos = (t_soft * (t_log - s_log)).sum(dim=-1)  # (B, T_d)
    valid_mask = (~decoder_mask).float()  # (B, T_d)
    # Per-clip T² scaling, then mask, then mean over valid positions.
    kl_per_pos = kl_per_pos * T_sq if isinstance(T_sq, torch.Tensor) else kl_per_pos * T_sq
    kl_per_pos = kl_per_pos * valid_mask
    kd_term = kl_per_pos.sum() / valid_mask.sum().clamp(min=1.0)

    # Hard target loss (CE against teacher's greedy tokens). decoder_mask is the
    # only correct mask: collate_fn zero-pads teacher_tokens, but pad_id from the
    # fairseq dict is typically 1 (not 0), so ignore_index=pad_id would mask
    # nothing and BOS would be supervised at every padded position.
    ce_per_pos = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_tokens.reshape(-1),
        reduction='none',
    ).reshape(student_logits.shape[:2])
    ce = (ce_per_pos * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    return alpha * kd_term + (1 - alpha) * ce, kd_term.item(), ce.item()


# ----- Viseme-conditioned learned temperatures -----

# Viseme bucketing matches Sebastian's analysis notebook exactly so our
# learned per-viseme temperatures line up with his fixed-temperature ablation.
# 14 categories with split vowels, plus 'other' for words not in CMUdict.
VISEME_LIST = [
    'bilabial', 'labiodental', 'dental', 'alveolar', 'postalveolar',
    'velar', 'glottal', 'approximant',
    'front_unrounded', 'open', 'rounded', 'diphthong', 'rhotic_vowel',
    'other',
]
N_VISEMES = len(VISEME_LIST)
VISEME_TO_IDX = {v: i for i, v in enumerate(VISEME_LIST)}

_PHONEME_TO_VISEME = {
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


def initial_viseme_idx(word, cmu):
    """Return viseme index for a word's first phoneme. Falls back to 'other'."""
    word = word.lower().strip()
    pron = cmu.get(word)
    if not pron:
        return VISEME_TO_IDX['other']
    first = pron[0][0]
    first = ''.join(c for c in first if not c.isdigit())
    return VISEME_TO_IDX.get(_PHONEME_TO_VISEME.get(first, 'other'),
                              VISEME_TO_IDX['other'])


def build_word_viseme_table(target_words_iter, cmu):
    """Map each unique target word to a viseme index. Returns a dict."""
    table = {}
    for w in target_words_iter:
        if w in table:
            continue
        table[w] = initial_viseme_idx(w, cmu)
    return table


def shift_right_for_teacher_forcing(tokens, bos_id, pad_id):
    """Convert (B, T) targets [t1, t2, ..., EOS] to decoder inputs [BOS, t1, t2, ...]"""
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
        
        # True KL (was previously CE, off by H(t) which doesn't affect argmin
        # but inflates the printed value). Correct formula: Σ t·(log t − log s).
        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
        kl_per_tok = (t_soft * (t_log - s_log)).sum(dim=-1)  # (B, T_d)
        valid = (~decoder_mask).float()
        total_kl += (kl_per_tok * valid).sum().item()
        total_tokens += valid.sum().item()

    return total_kl / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--ckpt', default='data/checkpoints/checkpoint.pt',
                    help='Teacher checkpoint, used only to get vocab and special token ids')
    ap.add_argument('--out_dir', default='runs/student')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--temperature', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='Mixing weight: alpha*KD + (1-alpha)*CE')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--enc_layers', type=int, default=4)
    ap.add_argument('--dec_layers', type=int, default=4)
    ap.add_argument('--early_stopping_patience', type=int, default=3,
                    help='Stop if val_kl does not improve for this many epochs.')
    ap.add_argument('--early_stopping_min_delta', type=float, default=1e-3,
                    help='Minimum val_kl improvement to count as a real improvement.')
    ap.add_argument('--learned_viseme_temp', action='store_true',
                    help='Learn one temperature per viseme category (10 params) '
                         'instead of using a single fixed temperature. Each clip\'s '
                         'target word maps to a viseme via CMUdict; the corresponding '
                         'log-temperature parameter scales that clip\'s KD loss. '
                         'Initialised so all temperatures equal --temperature.')
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Special token ids from teacher's dictionary
    print("Loading teacher dictionary for vocab info...")
    special = get_special_token_ids(args.ckpt)
    print(f"  vocab_size={special['vocab_size']} "
          f"pad={special['pad_id']} bos={special['bos_id']} eos={special['eos_id']}")
    
    # Datasets
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
    
    # Model
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
    print(f"Student: {n_params:.1f}M params")
    
    # Learnable viseme temperatures (the project's novel contribution).
    # Each viseme category gets one log-temperature scalar; learned jointly
    # with the model. Initialised so exp(log_temps[i]) == args.temperature for
    # all i, matching the uniform-T baseline at step 0.
    log_temps = None
    word_to_viseme = None
    if args.learned_viseme_temp:
        try:
            import cmudict
        except ImportError:
            raise SystemExit("--learned_viseme_temp requires `pip install cmudict`")
        cmu = cmudict.dict()
        # Pre-compute word -> viseme for every target word in train + val so
        # we never look up CMUdict in the hot loop.
        word_to_viseme = build_word_viseme_table(
            (w for ds in (train_set, val_set) for w in
             (cid.split('_')[0] for cid in ds.clip_ids)),
            cmu,
        )
        log_temps = torch.nn.Parameter(
            torch.full((N_VISEMES,), math.log(args.temperature),
                       device=device, dtype=torch.float32),
        )
        # Print the initial mapping summary
        from collections import Counter
        viseme_counts = Counter(word_to_viseme.values())
        print(f"  learnable viseme temps: {N_VISEMES} categories")
        for i, name in enumerate(VISEME_LIST):
            n = viseme_counts.get(i, 0)
            print(f"    {i}={name:<14} ({n} unique target words)")

    # Optimizer + scheduler
    train_params = list(model.parameters())
    if log_temps is not None:
        train_params.append(log_temps)
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_kl = float('inf')
    epochs_since_improvement = 0
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

            # Per-clip temperature: either uniform scalar, or per-clip tensor
            # derived from each target word's viseme category and the learnable
            # log_temps parameter.
            if log_temps is not None:
                viseme_idx = torch.tensor(
                    [word_to_viseme[w] for w in batch['target_words']],
                    device=device, dtype=torch.long,
                )  # (B,)
                clip_temp = torch.exp(log_temps[viseme_idx])  # (B,) — gradient flows through log_temps
            else:
                clip_temp = args.temperature

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                student_logits = model(video, video_mask, prev_tokens,
                                       decoder_mask=decoder_mask)
                loss, kd_val, ce_val = kd_loss(
                    student_logits.float(), teacher_logits, teacher_tokens,
                    decoder_mask,
                    temperature=clip_temp, alpha=args.alpha,
                    pad_id=special['pad_id'],
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
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
        
        print(f"epoch {epoch+1}: train_loss={train_loss:.3f} "
              f"val_kl(T={args.temperature})={val_kl:.3f} "
              f"time={elapsed:.0f}s")
        
        # If we're learning per-viseme temperatures, log them every epoch so we
        # can see how the model is choosing to soften per category over time.
        if log_temps is not None:
            with torch.no_grad():
                temps = torch.exp(log_temps).detach().cpu().tolist()
            print("  learned T per viseme:")
            for i, name in enumerate(VISEME_LIST):
                print(f"    T[{name:<14}] = {temps[i]:.3f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_kl': val_kl,
            'args': vars(args),
        }
        if log_temps is not None:
            ckpt['log_temps'] = log_temps.detach().cpu()
            ckpt['viseme_list'] = VISEME_LIST
            ckpt['word_to_viseme'] = word_to_viseme
        torch.save(ckpt, os.path.join(args.out_dir, 'last.pt'))
        if val_kl < best_val_kl - args.early_stopping_min_delta:
            best_val_kl = val_kl
            epochs_since_improvement = 0
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            print(f"  → new best val_kl={val_kl:.3f}, saved best.pt")
        else:
            epochs_since_improvement += 1
            print(f"  → no improvement ({epochs_since_improvement}/{args.early_stopping_patience} since best={best_val_kl:.3f})")
            if epochs_since_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}: val_kl plateaued for {args.early_stopping_patience} epochs.")
                break


if __name__ == '__main__':
    main()
