## Clone this repo
```
git clone https://github.com/seblini/slp-project.git
cd slp-project
```

## Using pyenv, and built-in python venv (alternatively, use conda. example in av-hubert repo)
```
pyenv install 3.8.0
pyenv local 3.8.0

python -m venv .venv
source .venv/bin/activate
```

## Clone av-hubert repo at the same level as this repo and install dependencies
```
cd ..
git clone https://github.com/facebookresearch/av_hubert.git
cd avhubert
git submodule init
git submodule update

pip install -r requirements.txt
cd fairseq
pip install --editable ./
```

## Downgrade numpy version to avoid problems with type aliases, https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```
pip install "numpy<1.24"
```

## Download the pretrained model: AV-HuBERT Base, LRS3 + VoxCeleb2 (EN), no finetuning
```
wget -O checkpoint.pt https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt
```
