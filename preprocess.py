# batch_preprocess.py with multiprocessing
import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(AV_HUBERT_PATH.resolve()))
sys.path.insert(0, str(FAIRSEQ_PATH.resolve()))


def find_videos(video_dir, ext='.mp4'):
    paths = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(ext):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def get_output_path(video_path, video_dir, output_dir):
    rel = os.path.relpath(video_path, video_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(output_dir, rel_no_ext + '.npz')


# Per-worker globals — initialized once per process
_extractor = None
_args = None


def init_worker(mean_face_path, video_dir, output_dir):
    import torch
    torch.set_num_threads(1)

    global _extractor, _args
    from roi import MouthROIExtractor
    _extractor = MouthROIExtractor(mean_face_path)
    _args = (video_dir, output_dir)


def process_one(video_path):
    global _extractor, _args
    video_dir, output_dir = _args
    
    out = get_output_path(video_path, video_dir, output_dir)
    if os.path.exists(out):
        return ('skipped', video_path)
    
    try:
        result = _extractor(video_path)
    except Exception as e:
        return ('error', video_path, str(e))
    
    if result is None or len(result['video']) < 4:
        return ('failed', video_path)
    
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, video=result['video'], audio=result['audio'])
    return ('ok', video_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--mean_face', default='data/misc/20words_mean_face.npy')
    ap.add_argument('--ext', default='.mp4')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()
    
    paths = find_videos(args.video_dir, args.ext)
    print(f"Found {len(paths)} videos")
    
    if args.skip_existing:
        before = len(paths)
        paths = [p for p in paths
                 if not os.path.exists(get_output_path(p, args.video_dir, args.output_dir))]
        print(f"Skipping {before - len(paths)} already-processed videos")
    
    failed = []
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.mean_face, args.video_dir, args.output_dir),
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_one, paths, chunksize=8),
                           total=len(paths)):
            if result[0] in ('failed', 'error'):
                failed.append(result[1])
    
    print(f"\nDone. Successfully processed {len(paths) - len(failed)}/{len(paths)}")
    if failed:
        with open('failed_videos.txt', 'w') as f:
            f.write('\n'.join(failed))
        print(f"Failed: {len(failed)}, see failed_videos.txt")


if __name__ == '__main__':
    set_start_method('spawn')  # CUDA + multiprocessing requires spawn
    main()
