"""Aggregate Sebastian's eval JSON by viseme category.

Sebastian's eval.py outputs `per_word_acc: {WORD: accuracy, ...}` keyed by LRW
class. This script loads one or more eval JSONs, looks up each word's initial-
phoneme viseme via CMUdict, and produces a per-viseme accuracy breakdown.

When given multiple JSONs, prints a side-by-side comparison table — exactly
what goes into the report's ablation section.

Usage:
    pip install nltk
    python -c "import nltk; nltk.download('cmudict')"
    python student/viseme_aggregator.py \\
        --eval runs/c2_hypA/eval_val.json \\
        --eval runs/c3_hypB/eval_val.json \\
        --eval runs/c4_learnable/eval_val.json \\
        --labels HypA HypB Learnable \\
        --output runs/SUMMARY.md
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Use NLTK's CMUdict (matches Sebastian's notebook).
try:
    import nltk
    from nltk.corpus import cmudict
    try:
        CMU = cmudict.dict()
    except LookupError:
        nltk.download('cmudict', quiet=True)
        CMU = cmudict.dict()
except ImportError:
    print("nltk not installed. Run: pip install nltk; python -c 'import nltk; nltk.download(\"cmudict\")'", file=sys.stderr)
    sys.exit(1)


# Sebastian's exact viseme mapping (from 02_analysis.ipynb).
# Each LRW word's INITIAL phoneme determines its viseme bucket. Initial-phoneme
# is the cleanest "where does the word START" signal — bilabial-initial words
# start with a lip closure (most ambiguous), labiodental-initial words show
# the distinctive teeth-on-lip gesture immediately, etc.
VISEME_MAP = {
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

# Print order: bilabial first (most ambiguous, the load-bearing comparison),
# then labiodental (least ambiguous), then alphabetical, with unknown last.
VISEME_ORDER = [
    'bilabial', 'labiodental', 'dental', 'alveolar', 'postalveolar',
    'velar', 'glottal', 'approximant', 'front_unrounded', 'open', 'rounded',
    'diphthong', 'rhotic_vowel', 'unknown',
]


def initial_viseme(word):
    """Return the viseme of the word's first phoneme, or 'unknown' if not in CMU."""
    word = word.lower().strip()
    pron = CMU.get(word)
    if not pron:
        return 'unknown'
    first = pron[0][0]
    first = ''.join(c for c in first if not c.isdigit())  # strip stress
    return VISEME_MAP.get(first, 'unknown')


def aggregate(per_word_acc):
    """Map word→accuracy into viseme→{n_words, mean_acc, words}."""
    by_viseme = defaultdict(lambda: {'words': [], 'accs': []})
    for word, acc in per_word_acc.items():
        v = initial_viseme(word)
        by_viseme[v]['words'].append(word)
        by_viseme[v]['accs'].append(acc)

    out = {}
    for v, d in by_viseme.items():
        out[v] = {
            'n_words': len(d['words']),
            'mean_acc': sum(d['accs']) / len(d['accs']) if d['accs'] else 0.0,
            'words': sorted(d['words']),
        }
    return out


def fmt_md_table(eval_summaries, labels):
    """Build a markdown comparison table of per-viseme word accuracy."""
    lines = []
    # Header
    headers = ['Viseme', 'n_words'] + labels
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    # Body
    for v in VISEME_ORDER:
        present = [s.get(v) for s in eval_summaries]
        if not any(p is not None for p in present):
            continue
        # n_words should be the same across runs (same dataset)
        n = next((p['n_words'] for p in present if p is not None), 0)
        row = [f'**{v}**' if v in ('bilabial', 'labiodental') else v, str(n)]
        for p in present:
            row.append(f"{p['mean_acc']*100:.1f}%" if p else '—')
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='append', required=True,
                    help='Path to eval JSON. Repeat for multiple conditions.')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='Display labels for each --eval, in same order.')
    ap.add_argument('--output', default=None,
                    help='Write markdown summary here. Stdout always gets the table.')
    args = ap.parse_args()

    if args.labels is not None and len(args.labels) != len(args.eval):
        raise SystemExit(f"--labels count ({len(args.labels)}) must match --eval count ({len(args.eval)}).")
    labels = args.labels or [Path(p).parent.name for p in args.eval]

    summaries = []
    top_metrics = []
    for path in args.eval:
        with open(path) as f:
            data = json.load(f)
        decoding = data.get('decoding', {})
        per_word = decoding.get('per_word_acc', {})
        if not per_word:
            print(f"WARN: {path} has no decoding.per_word_acc, skipping per-viseme table for it.", file=sys.stderr)
            summaries.append({})
        else:
            summaries.append(aggregate(per_word))

        top_metrics.append({
            'ckpt_epoch': data.get('ckpt_epoch'),
            'distribution_match': data.get('distribution_match', {}),
            'word_presence_acc': decoding.get('word_presence_acc'),
            'wer_vs_teacher': decoding.get('wer_vs_teacher'),
            'n_clips': decoding.get('n_clips'),
        })

    # Top-level metrics table
    md = []
    md.append('# Ablation results\n')
    md.append('## Top-level metrics\n')
    md.append('| Metric | ' + ' | '.join(labels) + ' |')
    md.append('|---' + '|---' * len(labels) + '|')
    rows = [
        ('ckpt_epoch', lambda m: m.get('ckpt_epoch'), '{:d}'),
        ('cross_entropy', lambda m: m['distribution_match'].get('cross_entropy'), '{:.4f}'),
        ('teacher_entropy', lambda m: m['distribution_match'].get('teacher_entropy'), '{:.4f}'),
        ('KL', lambda m: m['distribution_match'].get('kl'), '{:.4f}'),
        ('top1_agreement', lambda m: m['distribution_match'].get('top1_agreement'), '{:.4f}'),
        ('top5_agreement', lambda m: m['distribution_match'].get('top5_agreement'), '{:.4f}'),
        ('word_presence_acc', lambda m: m.get('word_presence_acc'), '{:.4f}'),
        ('wer_vs_teacher', lambda m: m.get('wer_vs_teacher'), '{:.4f}'),
        ('n_clips', lambda m: m.get('n_clips'), '{:d}'),
    ]
    for name, getter, fmt in rows:
        row = [name]
        for m in top_metrics:
            v = getter(m)
            row.append(fmt.format(v) if v is not None else '—')
        md.append('| ' + ' | '.join(row) + ' |')

    md.append('\n## Per-viseme word accuracy\n')
    md.append('Bilabials and labiodentals bolded — the load-bearing comparison '
              'for the Hyp A vs Hyp B question. Bilabial-initial words start with '
              'a lip closure (most ambiguous viseme); labiodentals start with '
              'teeth-on-lip (most distinctive).\n')
    md.append(fmt_md_table(summaries, labels))

    output = '\n'.join(md)
    print(output)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + '\n')
        print(f'\nWrote {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
