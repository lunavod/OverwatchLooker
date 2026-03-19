# OCR Tools

Tools for training and testing OCR models on the Overwatch 2 scoreboard font (Big Noodle Too Oblique).

## Setup

### Font
Download Big Noodle Too from [GitHub](https://github.com/Resike/Overwatch/tree/master/Fonts) and place in `fonts/`:
```
fonts/BigNoodleToo.ttf
fonts/BigNoodleTooOblique.ttf
```

### PaddleOCR GPU (Windows)

PaddlePaddle + PyTorch coexistence on Windows requires careful DLL management.

1. Install the setup tool (from a separate repo):
   ```
   pip install C:\path\to\paddleocr-gpu-setup
   paddleocr-gpu-setup
   ```
   Or manually:
   ```powershell
   # Install PaddlePaddle GPU from the official Chinese index
   uv pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

   # Install PaddleOCR
   uv pip install paddleocr

   # Remove conflicting nvidia pip packages (use system CUDA instead)
   uv pip uninstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

   # Sync libiomp5md.dll (copy torch's version into paddle)
   python -c "import shutil; from pathlib import Path; s=Path('.venv/Lib/site-packages'); shutil.copy2(s/'torch'/'lib'/'libiomp5md.dll', s/'paddle'/'libs'/'libiomp5md.dll')"
   ```

2. System CUDA DLLs must be on PATH at runtime:
   ```powershell
   $env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.8;" + $env:PATH
   ```

3. Install training dependencies:
   ```
   uv pip install albumentations lmdb rapidfuzz
   ```

### Tesseract (optional, for comparison)
Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) — includes training tools (`lstmtraining`, `combine_tessdata`, etc.).

## Training Data Generation

```bash
uv run python tools/generate_training_data.py --count 69000
```

Generates synthetic text line images in Big Noodle Too Oblique on OW2-style backgrounds:
- `.png` + `.gt.txt` pairs in `training_data/ow2-ground-truth/`
- Uppercase only (Big Noodle Too is a titling font — lowercase renders identical to uppercase)
- Latin + Cyrillic + accented characters
- Heavily weighted toward small sizes (18-28px) and digit-heavy content
- Real player names from MCP data (place `tools/mcp_player_names.txt`)

## PaddleOCR Fine-Tuning

1. Convert training data to PaddleOCR format:
   ```bash
   # This is done by generate_training_data.py or manually:
   uv run python -c "
   from pathlib import Path; import random
   # ... generates training_data/paddle_rec/{train_list.txt, val_list.txt, dict.txt}
   "
   ```

2. Clone PaddleOCR and download pretrained weights:
   ```bash
   git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git paddleocr_repo
   curl -sL 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams' -o training_data/PP-OCRv5_server_rec_pretrained.pdparams
   ```

3. Train (in a separate terminal with CUDA on PATH):
   ```powershell
   $env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.8;" + $env:PATH
   cd C:\Users\yegor\Projects\OverwatchLooker
   .venv\Scripts\python.exe paddleocr_repo\tools\train.py -c training_data\paddle_rec\ow2_finetune.yml
   ```
   Resume from checkpoint:
   ```powershell
   .venv\Scripts\python.exe paddleocr_repo\tools\train.py -c training_data\paddle_rec\ow2_finetune.yml -o Global.checkpoints=./training_data/paddle_rec/output/latest
   ```

4. Export trained model:
   ```bash
   uv run python paddleocr_repo/tools/export_model.py \
     -c training_data/paddle_rec/ow2_finetune.yml \
     -o Global.pretrained_model=./training_data/paddle_rec/output/best_accuracy.pdparams \
        Global.save_inference_dir=./training_data/paddle_rec/ow2_infer/
   ```
   Then fix the model name in `ow2_infer/inference.yml`: change `model_name: ow2_rec` to `model_name: PP-OCRv5_server_rec`.

5. Test:
   ```python
   from paddleocr import TextRecognition
   rec = TextRecognition(model_dir='training_data/paddle_rec/ow2_infer')
   result = rec.predict(input='usernames/LUNAVOD.png', batch_size=1)
   ```

### Training Results (100 epochs, 69K samples)

- Best accuracy: **91.12%** exact match (epoch 95)
- Character accuracy: **98.6%** (norm_edit_dis)
- Training time: ~9.5 hours on RTX 4090
- Usernames: 5/6 perfect on real screenshots (MC1R gets spurious space)

### Known Issues

- Single-digit stat crops (e.g. "4", "0") are too small for reliable recognition even with upscaling
- Stock PaddleOCR reads numbers better on full rows but can't read Big Noodle Too names
- Hybrid approach (fine-tuned for names, stock for numbers) needs better stat crop strategy
- Training on random backgrounds (not just OW2 colors) would improve robustness

## Scoreboard Slicing

```bash
uv run python tools/slice_scoreboard.py screenshot.png --debug debug_slicing/
```

Pipeline:
1. Color mask for ally blue / enemy red
2. Largest contour = scoreboard bounding box
3. Count rows via white text density peaks
4. Equal-slice into N rows
5. Trim portrait (scan for first all-background column, use enemy for both teams)
6. Cut ult charge area for ally (0.95 × row_height square)
7. Detect & filter perk/ult circles via HoughCircles
8. Group remaining white text blobs, OCR each

### Known Issues

- Ally ult charge with effects (full ult) can extend past the cut area
- Portrait cut uses enemy rows (clean) to avoid ult effects on ally rows
- Competitive rank badges may interfere with ally row detection
- Name + title groups need PSM 6 (block mode) not PSM 7 (single line)
