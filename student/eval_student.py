import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def shift_right_for_teacher_forcing(tokens, bos_id, pad_id):
    B, _ = tokens.shape
    bos = torch.full((B, 1), bos_id, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([bos, tokens[:, :-1]], dim=1)


def load_teacher_dict(teacher_ckpt_path):
    _, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([teacher_ckpt_path])
    d = task.target_dictionary
    return d, {
        'vocab_size': len(d),
        'pad_id': d.pad(),
        'bos_id': d.bos(),
        'eos_id': d.eos(),
    }


def trim_to_first_eos(token_ids, eos_id):
    out = []
    for t in token_ids:
        out.append(int(t))
        if int(t) == eos_id:
            break
    return out


@torch.no_grad()
def teacher_forced_metrics(model, loader, special, device, temperature):
    model.eval()
    bos = special['bos_id']
    pad = special['pad_id']

    total_ce = 0.0
    total_kl = 0.0
    total_argmax_match = 0
    total_top5_jaccard = 0.0
    total_valid = 0
    n_clips = 0

    for batch in loader:
        video = batch['video'].to(device, non_blocking=True)
        video_mask = batch['video_mask'].to(device, non_blocking=True)
        teacher_logits = batch['teacher_logits'].to(device, non_blocking=True)
        teacher_tokens = batch['teacher_tokens'].to(device, non_blocking=True)
        decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
        n_clips += video.shape[0]

        prev_tokens = shift_right_for_teacher_forcing(teacher_tokens, bos, pad)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            student_logits = model(video, video_mask, prev_tokens, decoder_mask=decoder_mask)

        # fp32 before softmax: fp16 softmax silently produces bad probabilities
        # for distributions with sharp peaks.
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        # decoder_mask (True = padded) is the only correct mask: the dataset
        # zero-pads teacher_tokens, but pad_id from the fairseq dict is usually
        # nonzero, so ignore_index-based masking would fail.
        valid = (~decoder_mask).float()
        n_valid = valid.sum().item()
        total_valid += int(n_valid)

        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        t_log = F.log_softmax(teacher_logits / temperature, dim=-1)

        ce_per_pos = -(t_soft * s_log).sum(dim=-1)
        total_ce += (ce_per_pos * valid).sum().item()

        kl_per_pos = (t_soft * (t_log - s_log)).sum(dim=-1)
        total_kl += (kl_per_pos * valid).sum().item()

        s_argmax = student_logits.argmax(dim=-1)
        t_argmax = teacher_logits.argmax(dim=-1)
        match = (s_argmax == t_argmax).float() * valid
        total_argmax_match += match.sum().item()

        s_top5 = student_logits.topk(5, dim=-1).indices
        t_top5 = teacher_logits.topk(5, dim=-1).indices
        inter = (s_top5.unsqueeze(-1) == t_top5.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        # |∪| = 10 - |∩| for two 5-element sets
        jaccard_per_pos = inter.float() / (10.0 - inter.float()).clamp(min=1.0)
        total_top5_jaccard += (jaccard_per_pos * valid).sum().item()

    if total_valid == 0:
        return {
            'n_clips': 0, 'n_valid_positions': 0,
            'ce_per_token': float('nan'), 'kl_per_token': float('nan'),
            'argmax_agreement': float('nan'), 'top5_jaccard': float('nan'),
        }

    return {
        'n_clips': n_clips,
        'n_valid_positions': total_valid,
        'ce_per_token': total_ce / total_valid,
        'kl_per_token': total_kl / total_valid,
        'argmax_agreement': total_argmax_match / total_valid,
        'top5_jaccard': total_top5_jaccard / total_valid,
    }


@torch.no_grad()
def greedy_autoregressive_metrics(model, dataset, special, device, max_decode_len,
                                  tgt_dict, batch_size, per_clip_records=None):
    """If per_clip_records is a list, per-clip records are appended for downstream
    analysis (e.g. viseme bucketing). Each record has clip_id, target_word, the
    decoded teacher/student texts, and boolean correctness flags."""
    model.eval()
    eos = special['eos_id']

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=True,
    )

    n_clips = 0
    n_string_match = 0
    sum_token_overlap = 0.0
    n_teacher_word_correct = 0
    n_student_word_correct = 0

    for batch in loader:
        video = batch['video'].to(device, non_blocking=True)
        video_mask = batch['video_mask'].to(device, non_blocking=True)
        teacher_tokens = batch['teacher_tokens']
        decoder_lens = batch['decoder_lens']
        clip_ids = batch['clip_ids']
        target_words = batch['target_words']

        with torch.cuda.amp.autocast(dtype=torch.float16):
            student_tokens_padded = model.greedy_decode(
                video, video_mask, max_len=max_decode_len,
            )
        student_tokens_padded = student_tokens_padded.cpu()

        for i in range(video.shape[0]):
            n_clips += 1

            t_len = int(decoder_lens[i].item())
            t_seq = trim_to_first_eos(teacher_tokens[i, :t_len].tolist(), eos)
            s_seq = trim_to_first_eos(student_tokens_padded[i].tolist(), eos)

            if s_seq == t_seq:
                n_string_match += 1

            if t_seq or s_seq:
                inter = sum((Counter(t_seq) & Counter(s_seq)).values())
                denom = max(len(t_seq), len(s_seq))
                if denom > 0:
                    sum_token_overlap += inter / denom

            # Word-accuracy: did the target word appear in the decoded transcript?
            # AV-HuBERT outputs sentencepiece subwords with ▁ word-boundary markers;
            # after replacing ▁ with space, words may also be split into subword pieces
            # separated by spaces (e.g. "mi n ist er" for "minister"). We strip ALL
            # spaces and substring-match against the lowercased target.
            t_text = tgt_dict.string(t_seq).replace('▁', ' ').strip().lower()
            s_text = tgt_dict.string(s_seq).replace('▁', ' ').strip().lower()
            target_compact = target_words[i].lower().strip().replace(' ', '')
            t_compact = t_text.replace(' ', '')
            s_compact = s_text.replace(' ', '')
            t_correct = bool(target_compact) and target_compact in t_compact
            s_correct = bool(target_compact) and target_compact in s_compact
            if t_correct:
                n_teacher_word_correct += 1
            if s_correct:
                n_student_word_correct += 1

            if per_clip_records is not None:
                per_clip_records.append({
                    'clip_id': clip_ids[i],
                    'target_word': target_words[i],
                    'teacher_text': t_text,
                    'student_text': s_text,
                    'teacher_correct': t_correct,
                    'student_correct': s_correct,
                })

    if n_clips == 0:
        return {
            'n_clips': 0,
            'greedy_string_match': float('nan'),
            'greedy_token_overlap': float('nan'),
            'teacher_word_accuracy': float('nan'),
            'student_word_accuracy': float('nan'),
        }

    return {
        'n_clips': n_clips,
        'greedy_string_match': n_string_match / n_clips,
        'greedy_token_overlap': sum_token_overlap / n_clips,
        'teacher_word_accuracy': n_teacher_word_correct / n_clips,
        'student_word_accuracy': n_student_word_correct / n_clips,
    }


def derive_label_word(clip_ids):
    """Pull the LRW class word(s) from clip IDs of form WORD_split_filename
    (or WORD1_WORD2_split_filename for compound dataset names). Returns the
    single label if there's exactly one class in the set, else None."""
    words = set()
    for cid in clip_ids:
        parts = cid.split('_')
        word_parts = []
        for p in parts:
            if p in ('train', 'val', 'test'):
                break
            word_parts.append(p)
        if word_parts:
            words.add(' '.join(word_parts))
    if len(words) == 1:
        return next(iter(words))
    return None


def format_report(report):
    tf = report['teacher_forced']
    g = report['greedy']
    lines = [
        f"slp-project student eval",
        f"  student: {report['student_ckpt']}",
        f"  data:    {report['videos']}",
        f"  logits:  {report['logits']}",
        f"  split:   {report['split']}",
        f"  T:       {report['temperature']}",
        "",
        f"Teacher-forced  ({tf['n_clips']} clips, {tf['n_valid_positions']} valid positions)",
        f"  ce_per_token       = {tf['ce_per_token']:.4f}",
        f"  kl_per_token       = {tf['kl_per_token']:.4f}",
        f"  argmax_agreement   = {tf['argmax_agreement']:.4f}",
        f"  top5_jaccard       = {tf['top5_jaccard']:.4f}",
        "",
        f"Greedy autoregressive  ({g['n_clips']} clips)",
        f"  greedy_string_match    = {g['greedy_string_match']:.4f}",
        f"  greedy_token_overlap   = {g['greedy_token_overlap']:.4f}",
        f"  teacher_word_accuracy  = {g['teacher_word_accuracy']:.4f}",
        f"  student_word_accuracy  = {g['student_word_accuracy']:.4f}",
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--teacher_ckpt', default='data/checkpoints/checkpoint.pt')
    ap.add_argument('--student_ckpt', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--max_decode_len', type=int, default=20)
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='1.0 = true distribution. Use 2.0 to compare against training KD logs.')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--output', default=None,
                    help='If set, write JSON report. Stdout always gets the human table.')
    ap.add_argument('--per_clip_output', default=None,
                    help='If set, write per-clip records to JSONL for downstream analysis (e.g. viseme_analysis.py).')
    ap.add_argument('--limit', type=int, default=None,
                    help='Optional: only evaluate first N clips of the split (smoke test).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    print(f"Loading teacher dictionary from {args.teacher_ckpt}")
    tgt_dict, special = load_teacher_dict(args.teacher_ckpt)
    print(f"  vocab_size={special['vocab_size']} "
          f"pad={special['pad_id']} bos={special['bos_id']} eos={special['eos_id']}")

    print(f"Loading student checkpoint {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    out_proj_vocab = state['out_proj.weight'].shape[0]
    if out_proj_vocab != special['vocab_size']:
        raise RuntimeError(
            f"Vocab mismatch: teacher dict has {special['vocab_size']} but "
            f"student out_proj has {out_proj_vocab}. Are you using the same "
            f"teacher checkpoint that was used during training?"
        )

    saved_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    dim = saved_args.get('dim', 256)
    enc_layers = saved_args.get('enc_layers', 4)
    dec_layers = saved_args.get('dec_layers', 4)

    model = VideoStudent(
        vocab_size=special['vocab_size'],
        dim=dim,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        pad_id=special['pad_id'],
        bos_id=special['bos_id'],
        eos_id=special['eos_id'],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Student loaded: {n_params:.1f}M params, dim={dim}, "
          f"enc_layers={enc_layers}, dec_layers={dec_layers}")

    print(f"Setting up {args.split} split")
    full = LRWDistillationDataset(args.videos, args.logits)
    splits = split_clip_ids_by_lrw_split(full.clip_ids)
    if args.split not in splits or not splits[args.split]:
        raise RuntimeError(f"Split '{args.split}' is empty or missing.")
    split_ids = splits[args.split]
    if args.limit:
        split_ids = split_ids[:args.limit]
    print(f"  {args.split}: {len(split_ids)} clips")

    eval_set = LRWDistillationDataset(args.videos, args.logits, clip_ids=split_ids)

    tf_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print("Running teacher-forced metrics...")
    tf_metrics = teacher_forced_metrics(
        model, tf_loader, special, device, temperature=args.temperature,
    )

    print("Running greedy autoregressive metrics...")
    per_clip_records = [] if args.per_clip_output else None
    greedy_metrics = greedy_autoregressive_metrics(
        model, eval_set, special, device,
        max_decode_len=args.max_decode_len,
        tgt_dict=tgt_dict,
        batch_size=min(8, args.batch_size),
        per_clip_records=per_clip_records,
    )

    report = {
        'student_ckpt': args.student_ckpt,
        'videos': args.videos,
        'logits': args.logits,
        'split': args.split,
        'temperature': args.temperature,
        'teacher_forced': tf_metrics,
        'greedy': greedy_metrics,
    }

    print()
    print(format_report(report))

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")

    if args.per_clip_output and per_clip_records is not None:
        out_dir = os.path.dirname(os.path.abspath(args.per_clip_output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.per_clip_output, 'w') as f:
            for rec in per_clip_records:
                f.write(json.dumps(rec) + '\n')
        print(f"Wrote {len(per_clip_records)} per-clip records to {args.per_clip_output}")


if __name__ == '__main__':
    main()
