import os
import h5py
import numpy as np
from tqdm import tqdm

ROI_DIR = 'data/lrw_roi'
OUT = 'data/lrw_pp_video/ABOUT_PRISON_pp_video.h5'

paths = []
for root, _, files in os.walk(ROI_DIR):
    for f in files:
        if f.endswith('.npz'):
            paths.append(os.path.join(root, f))
paths.sort()

def get_clip_id(npz_path, roi_dir):
    rel = os.path.relpath(npz_path, roi_dir)
    return os.path.splitext(rel)[0].replace('/', '_')

with h5py.File(OUT, 'w') as h5f:
    for npz in tqdm(paths):
        cid = get_clip_id(npz, ROI_DIR)
        try:
            data = np.load(npz)
            h5f.create_dataset(
                cid, data=data['video'],
                compression='gzip', compression_opts=4,
            )
        except Exception as e:
            print(f"  failed: {npz}: {e}")

print(f"Done. Saved videos to {OUT}")
