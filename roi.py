import face_alignment
import numpy as np
import av
import torch
from decord import VideoReader, cpu
import cv2
from preparation.align_mouth import landmarks_interpolate, crop_patch # type: ignore
import warnings
warnings.filterwarnings("ignore")

def extract_audio_16k(video_path, target_sr=16000):
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]
    
    # Set up resampler if needed
    src_rate = audio_stream.rate
    resampler = None
    if src_rate != target_sr:
        resampler = av.AudioResampler(format='flt', layout='mono', rate=target_sr)
    
    chunks = []
    for frame in container.decode(audio=0):
        if resampler:
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().flatten())
        else:
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)  # downmix to mono
            chunks.append(arr.flatten())
    
    container.close()
    return np.concatenate(chunks).astype(np.float32)

class MouthROIExtractor:
    def __init__(self, mean_face_path, device='cuda'):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
            face_detector='blazeface',
        )
        print('landmark net:', next(self.fa.face_alignment_net.parameters()).device)
        print('detector net:', next(self.fa.face_detector.face_detector.parameters()).device)
        print(type(self.fa.face_detector).__name__)
        self.mean_face = np.load(mean_face_path)
        self.stable_pts = [33, 36, 39, 42, 45]
        self.std_size = (256, 256)

    def __call__(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, 3) RGB
        except Exception as e:
            print(f"  decode failed for {video_path}: {e}")
            return None

        if len(frames) == 0:
            return None

        # Batched landmark detection
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        try:
            with torch.no_grad():
                preds = self.fa.get_landmarks_from_batch(tensor.cuda())
        except Exception as e:
            print(f"  landmark detection failed for {video_path}: {e}")
            return None

        landmarks = []
        for p in preds:
            if p is None or len(p) == 0:
                landmarks.append(None)
            else:
                # Single face assumption — take first
                lm = p[:68] if (hasattr(p, 'ndim') and p.ndim == 2) else p[0][:68]
                landmarks.append(lm)

        if all(lm is None for lm in landmarks):
            return None

        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            return None

        rois = crop_patch(
            video_path, landmarks, self.mean_face, self.stable_pts,
            STD_SIZE=self.std_size, window_margin=12,
            start_idx=48, stop_idx=68,
            crop_height=96, crop_width=96,
        )
        if rois is None:
            return None

        rois_gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rois], axis=0)

        try:
            audio = extract_audio_16k(video_path)
        except Exception as e:
            print(f"  audio extraction failed for {video_path}: {e}")
            return None
        
        return {'video': rois_gray.astype(np.uint8), 'audio': audio}
