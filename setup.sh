#!/usr/bin/env bash
# One-shot setup for a fresh RunPod (or other Linux+CUDA) box.
# Idempotent: safe to re-run. Sets up env, installs deps, downloads teacher
# checkpoint and mean face. Does NOT download LRW (needs your Oxford VGG creds)
# and does NOT run the pipeline. Prints next-step commands at the end.
#
# Usage:
#   bash setup.sh
#
# Override the workspace root by setting WORKSPACE before running.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SLP_DIR="$WORKSPACE/slp-project"
AVH_DIR="$WORKSPACE/av_hubert"
DATA_BASE="$SLP_DIR/data"

# Fork to clone (defaults to the seblini upstream; this script is committed
# inside Danielreyes2's fork, so override SLP_REPO_URL when calling if you want
# the fork's branch instead).
SLP_REPO_URL="${SLP_REPO_URL:-https://github.com/Danielreyes2/slp-project.git}"
SLP_BRANCH="${SLP_BRANCH:-baseline-with-eval}"

step() { echo; echo "===== $* ====="; }

step "System deps"
apt-get update
apt-get install -y python3.10 python3.10-venv python3-pip git wget tar build-essential

step "Clone repos into $WORKSPACE"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
if [ ! -d "$SLP_DIR/.git" ]; then
    git clone -b "$SLP_BRANCH" "$SLP_REPO_URL" "$SLP_DIR"
else
    echo "slp-project already cloned, skipping"
fi
if [ ! -d "$AVH_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/av_hubert.git "$AVH_DIR"
else
    echo "av_hubert already cloned, skipping"
fi

step "Python venv at $SLP_DIR/.venv"
cd "$SLP_DIR"
[ -d .venv ] || python3.10 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade "pip<24.1"

step "Install torch 2.7.1 + cu128"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

step "av_hubert + fairseq install"
cd "$AVH_DIR"
git submodule init && git submodule update
pip install -r requirements.txt --no-deps
cd "$AVH_DIR/fairseq"
pip install --editable ./ --no-deps
pip install "omegaconf==2.0.6" "hydra-core==1.0.7" --force-reinstall

step "fairseq numpy compat patch"
# Old fairseq uses np.float / np.int / np.bool / np.object which were removed
# in numpy 1.20+. Idempotent in practice (replacements are no-ops on rerun).
find . -name "*.py" -exec sed -i \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.int\b/np.int64/g' \
    -e 's/np\.bool\b/bool/g' \
    -e 's/np\.object\b/object/g' {} \;

step "fairseq checkpoint_utils.py weights_only patch"
# torch.load defaults changed; older fairseq calls torch.load without
# weights_only=False which raises on newer torch. README does this manually
# with nvim; we script it idempotently.
if grep -q 'weights_only=False' fairseq/checkpoint_utils.py; then
    echo "already patched"
else
    sed -i 's|state = torch\.load(f, map_location=torch\.device("cpu"))|state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)|' \
        fairseq/checkpoint_utils.py
fi

step "Preprocessing deps"
pip install scikit-video opencv-python decord h5py tqdm python_speech_features
pip install "face-alignment==1.3.5" --force-reinstall

step "Mean face + teacher checkpoint"
mkdir -p "$DATA_BASE/misc" "$DATA_BASE/checkpoints" \
         "$DATA_BASE/lrw_pp_video" "$DATA_BASE/lrw_logit"
[ -f "$DATA_BASE/misc/20words_mean_face.npy" ] || \
    wget -O "$DATA_BASE/misc/20words_mean_face.npy" \
        https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
[ -f "$DATA_BASE/checkpoints/checkpoint.pt" ] || \
    wget -O "$DATA_BASE/checkpoints/checkpoint.pt" \
        https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_433h.pt

step "Sanity: verify torch sees the GPU"
python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

cat <<EOF

===== Setup complete =====

Activate the venv in any new shell with:
    source $SLP_DIR/.venv/bin/activate

Next, run these manually (LRW credentials are yours, paste them in):

  # 1. Download LRW (~tens of GB)
  mkdir -p $DATA_BASE/lrw_mp4 && cd $DATA_BASE/lrw_mp4
  wget --user=YOUR_USER --password=YOUR_PW \\
      https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1.tar.bz2
  tar xjf lrw-v1.tar.bz2

  # 2. Subset to ABOUT and PRISON for the 2-class baseline
  mkdir -p $DATA_BASE/lrw_subset
  cp -r $DATA_BASE/lrw_mp4/lipread_mp4/ABOUT $DATA_BASE/lrw_subset/
  cp -r $DATA_BASE/lrw_mp4/lipread_mp4/PRISON $DATA_BASE/lrw_subset/

  # 3. Pipeline (cd $SLP_DIR first)
  cd $SLP_DIR
  python preprocessing/preprocess.py --video_dir data/lrw_subset --output_dir data/lrw_roi --workers 8
  python preprocessing/roi_to_video_h5.py
  python preprocessing/extract_logits.py --roi_dir data/lrw_roi --ckpt data/checkpoints/checkpoint.pt --output data/lrw_logit/ABOUT_PRISON_logits.h5 --batch_size 16
  python student/train_student.py --videos data/lrw_pp_video/ABOUT_PRISON_pp_video.h5 --logits data/lrw_logit/ABOUT_PRISON_logits.h5 --ckpt data/checkpoints/checkpoint.pt --out_dir runs/student_v1 --batch_size 32 --epochs 20
  python student/eval_student.py --videos data/lrw_pp_video/ABOUT_PRISON_pp_video.h5 --logits data/lrw_logit/ABOUT_PRISON_logits.h5 --teacher_ckpt data/checkpoints/checkpoint.pt --student_ckpt runs/student_v1/best.pt --split val --output runs/student_v1/eval_val.json
  python student/eval_student.py --videos data/lrw_pp_video/ABOUT_PRISON_pp_video.h5 --logits data/lrw_logit/ABOUT_PRISON_logits.h5 --teacher_ckpt data/checkpoints/checkpoint.pt --student_ckpt runs/student_v1/best.pt --split test --output runs/student_v1/eval_test.json

EOF
