# Training OCR Models

Guide for training or retraining the hero panel OCR models. All training, data generation, and export tools live in `everything_paddle/` — a self-contained uv subproject with its own Python environment and PaddlePaddle dependencies.

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA support (tested on RTX 4080)
- The fonts used by Overwatch 2's hero panel (not included in the repo — these are paid/free fonts, must be sourced by the user):
  - **Config Medium** (`Config-Medium.ttf`) — for stat labels. Available at [globalfonts.pro/font/config](https://globalfonts.pro/font/config).
  - **Futura No2 Demi Bold** (`Futura No2 Demi Bold.ttf` or similar) — for stat values. A community-sourced version is available at [Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts) (listed as `Futura No 2 D DemiBold.ttf`).
  - **Big Noodle Titling Oblique** (`big_noodle_titling_oblique.ttf` or similar) — for featured stat values. Available at [Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts).

## Setup

Run the init script from `everything_paddle/`:

```powershell
cd everything_paddle
.\init.ps1
```

This will:
1. Create a Python 3.11 environment with PaddlePaddle GPU, paddle2onnx, and all training deps (`uv sync`)
2. Clone PaddleOCR v3.1.1 into `paddleocr_repo/` (if not already present)
3. Check for the pretrained base model weights

If the pretrained weights are missing, download them manually from [PaddlePaddle/PP-OCRv5_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec) and place at `training_data/PP-OCRv5_server_rec_pretrained.pdparams`.

All commands below run from the `everything_paddle/` directory.

## Directory Structure

```
everything_paddle/
  pyproject.toml                 # PaddlePaddle + training dependencies
  init.ps1                      # One-time setup script
  export.py                     # ONNX export script
  tools/
    generate_label_training.py   # Synthetic data generator for labels
    generate_value_training.py   # Synthetic data generator for values
    generate_featured_training.py
    generate_delta_training.py
    generate_modifier_training.py
    generate_rank_training.py
    generate_side_training.py
    preprocess_recaug.py         # Offline RecAug augmentation
  training_data/
    PP-OCRv5_server_rec_pretrained.pdparams   # Base pretrained model
    panel_labels/                # Labels model training data + output
    panel_values/                # Values model training data + output
    panel_featured/              # Featured model training data + output
    ...
  paddleocr_repo/                # PaddleOCR clone (for training scripts)
```

## Step 1: Generate Synthetic Training Data

### Labels Model

```bash
uv run python tools/generate_label_training.py --font /path/to/Config-Medium.ttf --count 5000 --output training_data/panel_labels
```

Generates white-on-black text images using Config Medium font at varied sizes (24-44px, weighted toward 34px for 4K). Content is a mix of real OW2 stat label names (40%), random word combinations (30%), and gibberish (30%).

Output:
- `training_data/panel_labels/*.png` — training images
- `training_data/panel_labels/train_list.txt` — 90% train split
- `training_data/panel_labels/val_list.txt` — 10% val split
- `training_data/panel_labels/dict.txt` — character dictionary (A-Z + space)

### Values Model

```bash
uv run python tools/generate_value_training.py --font /path/to/Futura.ttf --count 5000 --output training_data/panel_values
```

Generates white-on-black number images using Futura font at varied sizes (32-60px, weighted toward 46px for 4K). Heavy oversampling of hard cases: zeros, standalone `1`, comma-separated numbers, timer patterns with colons.

Output: same structure as labels, with dict containing `0-9 % , . :`.

### Featured Model

```bash
uv run python tools/generate_featured_training.py --font /path/to/big_noodle_titling.ttf --count 5000 --output training_data/panel_featured
```

### Important: Keep It Simple

**Do NOT add color variations, background colors, or noise to the synthetic data.** Our experiments showed that clean white-on-black data with 5K samples achieves 100% accuracy on real game screenshots, while 69K samples with color variation achieved 0%. The model generalizes from clean renders to the game's semi-transparent panel because RecAug provides spatial robustness during training.

## Step 2: Write Training Config

Create a YAML config for PaddleOCR training. Full template (adjust paths and model-specific settings as noted):

```yaml
Global:
  model_name: panel_rec                                        # arbitrary name
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./training_data/<model>/output
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  calc_epoch_interval: 1
  pretrained_model: ./training_data/PP-OCRv5_server_rec_pretrained.pdparams
  checkpoints:                                                 # set to resume; clear before export
  save_inference_dir:
  use_visualdl: false
  character_dict_path: ./training_data/<model>/dict.txt
  max_text_length: &max_text_length 40                         # 40 for labels, 15 for values
  infer_mode: false
  use_space_char: true                                         # true for labels, false for values
  distributed: false
  save_res_path: ./training_data/<model>/output/predicts.txt
  d2s_train_image_shape: [3, 48, 320]

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.00005
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.0001

Architecture:
  model_type: rec
  algorithm: SVTR_HGNet
  Transform:
  Backbone:
    name: PPHGNetV2_B4
    text_rec: True
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./training_data/<model>/
    label_file_list:
    - ./training_data/<model>/train_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecAug:
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./training_data/<model>/
    label_file_list:
    - ./training_data/<model>/val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
```

Replace `<model>` with your model directory name (e.g. `panel_featured`).

**Key settings to adjust per model:**

| Setting | Labels | Values | Featured |
|---------|--------|--------|----------|
| `max_text_length` | 40 | 15 | 15 |
| `use_space_char` | true | false | false |
| `character_dict_path` | `panel_labels/dict.txt` | `panel_values/dict.txt` | `panel_featured/dict.txt` |

See `training_data/panel_featured_finetune.yml` for a complete working example.

## Step 2b: Pre-apply RecAug

RecAug (perspective warp, blur, noise, color inversion) is critical for generalization from synthetic to real data. Running it during training is very slow on Windows (CPU bottleneck doubles training time). Instead, pre-apply it offline:

```bash
uv run python tools/preprocess_recaug.py \
  --input training_data/panel_featured \
  --output training_data/panel_featured_aug \
  --variants 1 --workers 4
```

This creates augmented copies of the training images. The val images are NOT augmented (we want to evaluate on clean data).

**Important:** The training config must:
- Point `Train.dataset.data_dir` and `Train.dataset.label_file_list` at the `_aug` directory
- Point `Eval.dataset.data_dir` and `Eval.dataset.label_file_list` at the original (non-aug) directory
- NOT include `RecAug` in the transforms list (it's already been applied)
- Point `save_model_dir` and `save_res_path` at the original directory (output goes there)

**Do not skip RecAug.** Our v1 labels model trained with RecAug achieves 100% on real data; v2 trained without it (but with 69K pre-baked color variations) achieved 0%.

## Step 3: Train

```bash
uv run python -m paddleocr_repo.tools.train -c training_data/panel_featured_finetune.yml
```

### What to Expect

- **Epoch 1**: `acc: 0.0` is normal — the head layers reinitialize for the new character set while the backbone loads pretrained weights.
- **Epoch 2-5**: Accuracy climbs rapidly (90%+ by epoch 5 for small character sets).
- **Epoch 20-50**: Plateaus around 98-99%.
- **Epoch 100**: Final accuracy ~99.5% for labels, ~100% for values.

### Resuming from Checkpoint

If training is interrupted, set `checkpoints` in the config:

```yaml
Global:
  checkpoints: ./training_data/panel_values_v4/output/latest
```

Remove this line after training completes (it conflicts with export).

## Step 4: Export for Inference

After training, export the best checkpoint to a PaddlePaddle inference model:

```bash
uv run python -m paddleocr_repo.tools.export_model \
  -c training_data/panel_featured_finetune.yml \
  -o Global.checkpoints=./training_data/panel_featured/output/best_accuracy \
     Global.save_inference_dir=./training_data/panel_featured/inference
```

**Important**: Remove or clear the `checkpoints` field in the config before exporting, otherwise export will fail or load the wrong weights.

Fix the `model_name` field in the exported `inference.yml`:

```bash
sed -i 's/panel_featured_rec/PP-OCRv5_server_rec/' training_data/panel_featured/inference/inference.yml
```

## Step 5: Export to ONNX

Convert the PaddlePaddle inference model to ONNX format. The main app uses ONNX Runtime for inference — no PaddlePaddle at runtime.

Export all models at once:

```bash
uv run python export.py
```

Or export a specific model:

```bash
uv run python export.py training_data/panel_featured/inference
```

Verify the output `.onnx` file is ~72 MB (not 0 bytes).

## Step 6: Deploy

Copy the exported ONNX model and character dictionary to the production location:

```bash
mkdir -p ../overwatchlooker/models/<name>
cp training_data/<name>/inference/inference.onnx ../overwatchlooker/models/<name>/
cp training_data/<name>/dict.txt ../overwatchlooker/models/<name>/
```

Or for training-only models (rank_division, modifiers):

```bash
uv run python export.py --deploy
```

Only `inference.onnx` and `dict.txt` are needed at runtime.

## Lessons Learned

These are hard-won findings from multiple training iterations. Read before changing anything.

### What Works

- **5K clean white-on-black samples** — sufficient for 100% real-world accuracy with restricted character sets
- **RecAug** — provides spatial robustness (perspective, crop, distortion) that bridges the synthetic-to-real gap. Pre-apply offline with `tools/preprocess_recaug.py` for faster training
- **Restricted character sets** — A-Z+space for labels, 0-9+%+,+.+: for values. Eliminates O/0 and I/1 confusion entirely
- **Separate model per font** — the game uses different fonts for different text elements (Config Medium, Futura, Big Noodle Titling). Each model learns one font's shapes perfectly
- **~35 minutes training on RTX 4080** — 5K samples, 100 epochs, 99.8%+ accuracy (~10 min to first good checkpoint)

### What Does NOT Work

- **Color/background variation in training data** — 69K samples with varied colors scored 99.5% on synthetic data but **0% on real game screenshots**. The model overfits to PIL's font rendering artifacts instead of learning robust letterforms.
- **Removing RecAug** — without spatial augmentation the model doesn't generalize from synthetic to real, even with more training data
- **Feeding full-width strips to the values model** — 869px strips with a single thin character (like `1`) get destroyed when resized to 320px. Always crop to text bounds first.

## Quick Reference: Adding a New Model

All commands run from `everything_paddle/`. Replace `<name>` with your model name.

```bash
# 1. Create training data generator (copy and adapt an existing one)
cp tools/generate_value_training.py tools/generate_<name>_training.py

# 2. Create training config
cp training_data/panel_featured_finetune.yml training_data/<name>_finetune.yml

# 3. Generate training data
uv run python tools/generate_<name>_training.py --font /path/to/font.ttf

# 4. Pre-apply RecAug
uv run python tools/preprocess_recaug.py \
  --input training_data/<name> \
  --output training_data/<name>_aug \
  --variants 1 --workers 4

# 5. Train
uv run python -m paddleocr_repo.tools.train -c training_data/<name>_finetune.yml

# 6. Export PaddlePaddle inference model
uv run python -m paddleocr_repo.tools.export_model \
  -c training_data/<name>_finetune.yml \
  -o Global.checkpoints=./training_data/<name>/output/best_accuracy \
     Global.save_inference_dir=./training_data/<name>/inference

# 7. Fix model name
sed -i 's/<name>_rec/PP-OCRv5_server_rec/' training_data/<name>/inference/inference.yml

# 8. Export to ONNX
uv run python export.py training_data/<name>/inference

# 9. Deploy to production
cp training_data/<name>/inference/inference.onnx ../overwatchlooker/models/<name>/
cp training_data/<name>/dict.txt ../overwatchlooker/models/<name>/
```
