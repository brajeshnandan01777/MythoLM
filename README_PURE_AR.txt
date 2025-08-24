
MythoLM (Pure Autoregressive Transformer)

This version trains a decoder-only transformer **from scratch** (no Hugging Face models).
It uses a **byte-level vocabulary (0..255)** so it can model any UTF-8 text without tokenizers.

SETUP GUIDE
----------
# 0)Github create -> code .

conda create -p venv python==3.8 -y
conda activate venv/

git init
create README.md manually
git commit -m "first commit"
git branch -m main
git remote add origin Https://github.com/brajeshnandan01777/MythoLM
git push -u origin main
create .gitignore from github
git pull

# Install PyTorch with CUDA that matches your driver (example for CUDA 12.1):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

pip install -r requirements.txt

pip install requests tqdm
(A Python HTTP library used to fetch data from the internet ->requests
A progress bar library(wraps around loops so you can see how far along a task is)->tqdm)

pip install beautifulsoup4 lxml
(beautifulsoup4 → HTML parser
lxml → faster backend parser (optional, but recommended))


# 1) Build dataset (downloads public-domain myths)
python scripts/prepare_myth_dataset.py --out_dir data


# 2) Train on GPU
python scripts/train_pure_ar.py --train data/train.txt --val data/val.txt --out_dir checkpoints/pure_ar_mytholm --epochs 3 --max_seq 1024 --batch_size 8

watch -n 1 nvidia-smi(to see the updates of gpu usage and model training)
 
 Train — Fast / Safe config:-
 python scripts/train_pure_ar.py \
  --train data/train.txt \
  --val data/val.txt \
  --out_dir checkpoints/pure_ar_mytholm_fast \
  --n_embd 256 --n_head 4 --n_layer 4 \
  --max_seq 512 \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 4 \
  --eval_every 1000 \
  --grad_clip 1.0

# 3) Generate from best checkpoint:-
python scripts/generate_pure_ar.py \
  --ckpt checkpoints/pure_ar_mytholm_balanced/best.pt \
  --prompt "Sing, O Muse, of the deeds of the heroes," \
  --max_new_tokens 300 \
  --temperature 0.9 --top_p 0.9


