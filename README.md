Clone this repo
```
git clone https://github.com/seblini/slp-project.git
cd slp-project
```

Using pyenv, and built-in python venv (alternatively, use conda. example in av-hubert repo)
```
pyenv install 3.10.4
pyenv local 3.10.4

python -m venv .venv
source .venv/bin/activate
pip install "pip<24.1"
```

Install torch (cu128 is for nvidia 50 series, may need to change for other cards)
```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Clone av-hubert repo at the same level as this repo and install dependencies
```
cd ..
git clone https://github.com/facebookresearch/av_hubert.git

cd avhubert
git submodule init && git submodule update
pip install -r requirements.txt --no-deps

cd fairseq
pip install --editable ./ --no-deps
pip install "omegaconf==2.0.6" "hydra-core==1.0.7" --force-reinstall
```

Preprocessing dependencies
```
pip install scikit-video opencv-python decord h5py tqdm
pip install "face-alignment==1.3.5" --force-reinstall
```

Patch fairseq for 3.10.4 compatibility
```
find . -name "*.py" -exec sed -i \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.int\b/np.int64/g' \
    -e 's/np\.bool\b/bool/g' \
    -e 's/np\.object\b/object/g' \
    {} \;

nvim fairseq/checkpoint_utils.py
# Replace line 304 with:
# state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)

cd ..
```

Donwload mean face for landmarking
```
mkdir -p data/misc
wget -O data/misc/20words_mean_face.npy https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
```

Download the model: Noise-Augmented AV-HuBERT Large, LRS3 + VoxCeleb2 (EN), LRS3-422h
```
mkdir -p data/checkpoints
wget -O data/checkpoints/checkpoint.pt https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_433h.pt
```

Donwload the LRW dataset
```
mkdir -p data/lrw_mp4
wget https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1.tar.bz2
tar xvjf lrw-v1.tar.bz2
mv lip_reading lrw_mp4
```



