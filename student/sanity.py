import sys
import torch
from pathlib import Path
import h5py

AVHUBERT_PARENT = (Path('.') / "../av_hubert").resolve()
AVHUBERT_DIR = (Path('.') / "../av_hubert/avhubert").resolve()
FAIRSEQ_PATH = (Path('.') / "../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(AVHUBERT_DIR))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq
import hubert_pretraining, hubert, hubert_asr  # noqa
from student_model import VideoStudent
from student_dataset import LRWDistillationDataset

# Load teacher dictionary
_, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    ['data/checkpoints/checkpoint.pt'])
d = task.target_dictionary

# Load student
ckpt = torch.load('runs/student_v1/best.pt', map_location='cuda', weights_only=False)
student = VideoStudent(
    vocab_size=len(d),
    pad_id=d.pad(), bos_id=d.bos(), eos_id=d.eos(),
).cuda().eval()
student.load_state_dict(ckpt['model'])

# Load some test clips
ds = LRWDistillationDataset(
    'data/lrw_pp_video/ABOUT_PRISON_pp_video.h5',
    'data/lrw_logit/ABOUT_PRISON_logits.h5',
)

# Pick 10 random val clips
import random
random.seed(0)
val_ids = [c for c in ds.clip_ids if c.split('_')[1] == 'val']
samples = random.sample(val_ids, 10)

print(f"{'clip':<35} {'target':<10} {'teacher':<35} {'student'}")
print("-" * 110)
for cid in samples:
    item_idx = ds.clip_ids.index(cid)
    item = ds[item_idx]
    
    video = item['video'].unsqueeze(0).cuda()
    mask = torch.zeros(1, video.shape[1], dtype=torch.bool, device='cuda')
    
    with torch.no_grad():
        student_tokens = student.greedy_decode(video, mask, max_len=20)
    
    teacher_text = d.string(item['teacher_tokens'].tolist()).replace('▁', ' ').strip()
    student_text = d.string(student_tokens[0].tolist()).replace('▁', ' ').strip()
    
    print(f"{cid:<35} {item['target_word']:<10} {teacher_text:<35} {student_text}")
