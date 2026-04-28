import argparse
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
AVHUBERT_DIR = AVHUBERT_PARENT / "avhubert"
FAIRSEQ_PATH = (Path(__file__).parent / "../av_hubert/fairseq").resolve()
# AVHUBERT_DIR also on sys.path so avhubert/hubert.py's bare
# `from hubert_pretraining import ...` resolves (it's a sibling file, not relative).
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(AVHUBERT_DIR))
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
            temperature=2.0, alpha=0.5, pad_id=1):
    """
    Combined distillation loss:
      KD term: KL(student || teacher) at temperature T
      CE term: cross-entropy on teacher's greedy tokens (hard targets)
      total = alpha * KD + (1 - alpha) * CE
    """
    # student_logits, teacher_logits: (B, T_d, V)
    # teacher_tokens: (B, T_d)
    # decoder_mask: (B, T_d), True = padded
    
    valid = (~decoder_mask).float().unsqueeze(-1)  # (B, T_d, 1)
    
    # Soft target loss (KL with temperature)
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl = -(t_soft * s_log).sum(dim=-1)  # (B, T_d)
    kl = kl * (~decoder_mask).float()
    kd_term = kl.sum() / (~decoder_mask).float().sum().clamp(min=1.0)
    kd_term = kd_term * (temperature ** 2)  # standard KD scaling
    
    # Hard target loss (CE against teacher's greedy tokens). decoder_mask is the
    # only correct mask: collate_fn zero-pads teacher_tokens, but pad_id from the
    # fairseq dict is typically 1 (not 0), so ignore_index=pad_id would mask
    # nothing and BOS would be supervised at every padded position.
    ce_per_pos = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_tokens.reshape(-1),
        reduction='none',
    ).reshape(student_logits.shape[:2])
    ce_valid = (~decoder_mask).float()
    ce = (ce_per_pos * ce_valid).sum() / ce_valid.sum().clamp(min=1.0)
    
    return alpha * kd_term + (1 - alpha) * ce, kd_term.item(), ce.item()


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
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
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
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                student_logits = model(video, video_mask, prev_tokens,
                                       decoder_mask=decoder_mask)
                loss, kd_val, ce_val = kd_loss(
                    student_logits.float(), teacher_logits, teacher_tokens,
                    decoder_mask,
                    temperature=args.temperature, alpha=args.alpha,
                    pad_id=special['pad_id'],
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        
        # Save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_kl': val_kl,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, 'last.pt'))
        if val_kl < best_val_kl:
            best_val_kl = val_kl
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            print(f"  → new best val_kl={val_kl:.3f}, saved best.pt")


if __name__ == '__main__':
    main()
