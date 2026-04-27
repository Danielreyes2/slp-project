import numpy as np
import cv2
import argparse
import subprocess
import tempfile
import os


def rois_to_video(npz_path, output_path, fps=25, with_audio=True):
    """Render saved .npz back to a playable mp4 with audio."""
    data = np.load(npz_path)
    video = data['video']  # (T, 96, 96) uint8 grayscale
    
    h, w = video.shape[1], video.shape[2]
    
    # Write video-only first (silent mp4)
    silent_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (w, h), isColor=False)
    for frame in video:
        writer.write(frame)
    writer.release()
    
    if not with_audio or 'audio' not in data:
        os.replace(silent_path, output_path)
        return
    
    # Save audio to temp wav
    audio = data['audio']
    wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    import soundfile as sf
    sf.write(wav_path, audio, 16000)
    
    # Mux video + audio with ffmpeg
    subprocess.run([
        'ffmpeg', '-y',
        '-i', silent_path,
        '-i', wav_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        '-loglevel', 'error',
        output_path,
    ], check=True)
    
    os.unlink(silent_path)
    os.unlink(wav_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz_path')
    ap.add_argument('--output', default='roi_check.mp4')
    ap.add_argument('--no_audio', action='store_true')
    args = ap.parse_args()
    
    rois_to_video(args.npz_path, args.output, with_audio=not args.no_audio)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
