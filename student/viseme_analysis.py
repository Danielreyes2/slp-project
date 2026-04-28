"""Bucket per-clip eval results by initial viseme of the target word.

Reads the JSONL produced by `eval_student.py --per_clip_output`, groups clips by
the target word's initial-phoneme viseme category (looked up via CMUdict), and
prints word-accuracy per category. Optional JSON summary output.

The viseme grouping is the project's secondary metric — bilabial vs labiodental
vs other place-of-articulation categories. Bilabials (/b/, /p/, /m/) are
expected to be the hardest because they share an identical visible mouth
movement (lip closure). Labiodentals (/f/, /v/) are expected to be among the
easiest because the teeth-on-lip gesture is highly distinctive.

Usage:
    pip install cmudict
    python student/viseme_analysis.py \\
        --records runs/baseline_T2/eval_test_clips.jsonl \\
        --output  runs/baseline_T2/viseme_summary.json
"""
import argparse
import json
import sys
from collections import defaultdict

try:
    import cmudict
    CMU = cmudict.dict()
except ImportError:
    print("cmudict not installed. Run: pip install cmudict", file=sys.stderr)
    sys.exit(1)


# ARPAbet phoneme → viseme category, grouped by place of articulation.
# Categories follow the project doc's IPA-style groupings; the bilabial vs
# labiodental contrast is the load-bearing comparison for the temperature
# ablation hypotheses.
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
    'R': 'liquid_glide', 'Y': 'liquid_glide', 'W': 'liquid_glide',
    'AA': 'vowel', 'AE': 'vowel', 'AH': 'vowel', 'AO': 'vowel',
    'AW': 'vowel', 'AY': 'vowel', 'EH': 'vowel', 'ER': 'vowel',
    'EY': 'vowel', 'IH': 'vowel', 'IY': 'vowel', 'OW': 'vowel',
    'OY': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
}


def initial_viseme(word):
    """Return the initial-phoneme viseme category for a word, or 'unknown' if
    the word isn't in CMUdict."""
    word = word.lower().strip()
    pron = CMU.get(word)
    if not pron:
        return 'unknown'
    first_phone = pron[0][0]
    # ARPAbet vowels carry stress digits (AH0, EH1, ...); strip them.
    first_phone_no_stress = ''.join(c for c in first_phone if not c.isdigit())
    return PHONEME_TO_VISEME.get(first_phone_no_stress, 'unknown')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True,
                    help='Per-clip JSONL from eval_student.py --per_clip_output')
    ap.add_argument('--output', default=None,
                    help='If set, write per-viseme summary as JSON.')
    args = ap.parse_args()

    by_viseme = defaultdict(lambda: {'n': 0, 'student_correct': 0, 'teacher_correct': 0})
    word_to_viseme = {}

    with open(args.records) as f:
        for line in f:
            r = json.loads(line)
            tw = r['target_word']
            if tw not in word_to_viseme:
                word_to_viseme[tw] = initial_viseme(tw)
            v = word_to_viseme[tw]
            b = by_viseme[v]
            b['n'] += 1
            if r['student_correct']:
                b['student_correct'] += 1
            if r['teacher_correct']:
                b['teacher_correct'] += 1

    # Pretty table
    header = f"{'viseme':<15} {'n_clips':>8} {'unique_w':>8} {'student':>10} {'teacher':>10} {'gap':>+10}"
    print()
    print(header)
    print('-' * len(header))

    summary = {}
    # Group: bilabials and labiodentals first (the load-bearing comparison),
    # then everything else alphabetically, with unknown last.
    priority = {'bilabial': 0, 'labiodental': 1}
    keys = sorted(
        by_viseme.keys(),
        key=lambda v: (priority.get(v, 100 if v != 'unknown' else 999), v),
    )

    for v in keys:
        b = by_viseme[v]
        s_acc = b['student_correct'] / b['n']
        t_acc = b['teacher_correct'] / b['n']
        gap = t_acc - s_acc
        unique_w = sum(1 for w, vw in word_to_viseme.items() if vw == v)
        print(f"{v:<15} {b['n']:>8} {unique_w:>8} {s_acc:>10.4f} {t_acc:>10.4f} {gap:>+10.4f}")
        summary[v] = {
            'n_clips': b['n'],
            'unique_words': unique_w,
            'student_word_accuracy': s_acc,
            'teacher_word_accuracy': t_acc,
            'gap_teacher_minus_student': gap,
        }

    total_n = sum(b['n'] for b in by_viseme.values())
    total_s = sum(b['student_correct'] for b in by_viseme.values())
    total_t = sum(b['teacher_correct'] for b in by_viseme.values())
    print('-' * len(header))
    print(f"{'OVERALL':<15} {total_n:>8} {len(word_to_viseme):>8} "
          f"{total_s/total_n:>10.4f} {total_t/total_n:>10.4f} {(total_t-total_s)/total_n:>+10.4f}")

    summary['_overall'] = {
        'n_clips': total_n,
        'unique_words': len(word_to_viseme),
        'student_word_accuracy': total_s / total_n,
        'teacher_word_accuracy': total_t / total_n,
        'gap_teacher_minus_student': (total_t - total_s) / total_n,
    }

    # Print which words went into each category for sanity-check / reproducibility
    print("\nTarget words by viseme (first 15 per category):")
    word_groups = defaultdict(list)
    for w, v in word_to_viseme.items():
        word_groups[v].append(w)
    for v in keys:
        ws = sorted(word_groups[v])
        more = '...' if len(ws) > 15 else ''
        print(f"  {v}: {', '.join(ws[:15])}{more}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
