# Training OCR Models

Guide for training or retraining the hero panel OCR models from scratch on a freshly cloned repository.

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA support (tested on RTX 4080)
- PaddlePaddle GPU and PaddleOCR installed (`uv sync` handles this)
- The fonts used by Overwatch 2's hero panel (not included in the repo — these are paid/free fonts, must be sourced by the user):
  - **Config Medium** (`Config-Medium.ttf`) — for stat labels. Available at [globalfonts.pro/font/config](https://globalfonts.pro/font/config).
  - **Futura No2 Demi Bold** (`Futura No2 Demi Bold.ttf` or similar) — for stat values. A community-sourced version is available at [Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts) (listed as `Futura No 2 D DemiBold.ttf`).
  - **Big Noodle Titling Oblique** (`big_noodle_titling_oblique.ttf` or similar) — for featured stat values. Available at [Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts).

## Repository Structure

```
tools/
  generate_label_training.py    # Synthetic data generator for labels
  generate_value_training.py    # Synthetic data generator for values
  generate_featured_training.py # Synthetic data generator for featured values
  preprocess_recaug.py          # Offline RecAug augmentation
training_data/
  PP-OCRv5_server_rec_pretrained.pdparams   # Base pretrained model (required)
  panel_labels/                 # Labels model training data + output
  panel_values/                 # Values model training data + output
  panel_featured/               # Featured model training data + output
  panel_featured_finetune.yml   # Example training config (featured model)
paddleocr_repo/                 # PaddleOCR clone (used for training scripts)
overwatchlooker/models/         # Production inference models (git-tracked)
  panel_labels/                 # Exported labels model
  panel_values/                 # Exported values model
  panel_featured/               # Exported featured values model
```

## Step 0: Setup Dependencies

### Clone PaddleOCR

The training scripts live in a local PaddleOCR clone (gitignored):

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git paddleocr_repo
```

### Get the Pretrained Base Model

The PP-OCRv5 server rec pretrained weights are required as the starting point for finetuning. Download from [PaddlePaddle/PP-OCRv5_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec) and place at:

```
training_data/PP-OCRv5_server_rec_pretrained.pdparams
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

Generates white-on-black number images using Big Noodle Titling font at larger sizes (48-96px, weighted toward 80px for 4K). Heavy oversampling of timer patterns (MM:SS) since the featured stat frequently shows objective contest time.

Output: same structure as values, with dict containing `0-9 % , . :`.

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
    learning_rate: 0.00005                                     # 5e-5, good for single-GPU
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
    - RecAug:                                                  # KEEP — critical for generalization
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

### Training Speed

With RecAug and `num_workers: 4` on a single GPU:
- 5K samples: ~20-30 minutes for 100 epochs
- 69K samples: ~6-7 hours for 100 epochs

Without RecAug: roughly half the time, but **do not do this** for production models.

### Resuming from Checkpoint

If training is interrupted, set `checkpoints` in the config:

```yaml
Global:
  checkpoints: ./training_data/panel_values_v4/output/latest
```

Remove this line after training completes (it conflicts with export).

## Step 4: Export for Inference

After training, export the best checkpoint to an inference model:

```bash
uv run python -m paddleocr_repo.tools.export_model \
  -c training_data/panel_featured_finetune.yml \
  -o Global.checkpoints=./training_data/panel_featured/output/best_accuracy \
     Global.save_inference_dir=./training_data/panel_featured/inference
```

**Important**: Remove or clear the `checkpoints` field in the config before exporting, otherwise export will fail or load the wrong weights.

The exported model has a `model_name` field in `inference.yml` that must match `PP-OCRv5_server_rec` for `paddlex.create_model` to load it. Fix it:

```bash
sed -i 's/panel_featured_rec/PP-OCRv5_server_rec/' training_data/panel_featured/inference/inference.yml
```

The sed pattern is always `s/<config_model_name>/PP-OCRv5_server_rec/` where `<config_model_name>` is the `model_name` from your training config's `Global` section.

## Step 5: Test on Real Screenshots

Test the exported model directly:

```python
from paddlex import create_model
import cv2

model = create_model(model_name='PP-OCRv5_server_rec',
                     model_dir='training_data/panel_featured/inference')
img = cv2.imread('path/to/cropped_value.png')
result = list(model.predict(img))
print(result[0]["rec_text"], result[0]["rec_score"])
```

Or run the full hero panel pipeline:

```bash
uv run python debug_panel_structure.py path/to/tab_screenshot.png
```

Expected output: every label and value read correctly. If not, check:

1. **Values read as empty or garbled**: Are you cropping to text bounds before feeding to the model? Full 869px strips must be cropped to just the text region.
2. **Labels misread**: Check that RecAug was applied (either during training or via preprocessing).
3. **Commas read as periods**: Ensure `,` is in the character dictionary and training data has comma-number samples.
4. **Featured value wrong**: Check the font — the game uses a different font for the featured stat than for regular values.

## Step 6: Deploy

Copy the exported inference model to the production location:

```bash
cp training_data/panel_featured/inference/* overwatchlooker/models/panel_featured/
cp training_data/panel_featured/dict.txt overwatchlooker/models/panel_featured/
```

Commit via git (`.pdiparams` files are tracked via Git LFS).

Then wire it into `hero_panel.py`:
1. Add a lazy-loaded model function (`_get_featured_model()`) following the pattern of `_get_values_model()`
2. Add it to `preload_models()`
3. Use it in the appropriate OCR function

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
- **Large training sets without RecAug** — more data doesn't help if the model can't bridge the rendering gap
- **Feeding full-width strips to the values model** — 869px strips with a single thin character (like `1`) get destroyed when resized to 320px. Always crop to text bounds first.

### Architecture Reference

All models use the same architecture (finetuned from PP-OCRv5 server rec):

```
Backbone: PPHGNetV2_B4 (text_rec mode)
Neck: SVTR (dims=120, depth=2, hidden_dims=120)
Head: MultiHead
  - CTCHead (with SVTR neck, use_guide=True)
  - NRTRHead (nrtr_dim=384)
Loss: MultiLoss (CTCLoss + NRTRLoss)
PostProcess: CTCLabelDecode
Input shape: [3, 48, 320]
```

## Quick Reference: Adding a New Model

Copy-paste checklist for adding a new OCR model for a different font. Replace `<name>` with your model name (e.g. `panel_featured`).

```bash
# 1. Create training data generator (copy and adapt an existing one)
cp tools/generate_value_training.py tools/generate_<name>_training.py
# Edit: change font sizes, value distribution, output prefix, default output dir

# 2. Create training config (copy the featured model config as template)
cp training_data/panel_featured_finetune.yml training_data/<name>_finetune.yml
# Edit: replace panel_featured with <name>, adjust max_text_length/use_space_char

# 3. Generate training data
uv run python tools/generate_<name>_training.py --font /path/to/font.ttf

# 4. Pre-apply RecAug
uv run python tools/preprocess_recaug.py \
  --input training_data/<name> \
  --output training_data/<name>_aug \
  --variants 1 --workers 4

# 5. Train (~35 min on RTX 4080, first good checkpoint ~10 min)
uv run python -m paddleocr_repo.tools.train -c training_data/<name>_finetune.yml

# 6. Export
uv run python -m paddleocr_repo.tools.export_model \
  -c training_data/<name>_finetune.yml \
  -o Global.checkpoints=./training_data/<name>/output/best_accuracy \
     Global.save_inference_dir=./training_data/<name>/inference

# 7. Fix model name in exported config
sed -i 's/<name>_rec/PP-OCRv5_server_rec/' training_data/<name>/inference/inference.yml

# 8. Deploy to production
mkdir -p overwatchlooker/models/<name>
cp training_data/<name>/inference/* overwatchlooker/models/<name>/
cp training_data/<name>/dict.txt overwatchlooker/models/<name>/
```

Training config checklist:
- `Train` data_dir + label_file_list → `<name>_aug/` (augmented)
- `Eval` data_dir + label_file_list → `<name>/` (original, clean)
- `save_model_dir` + `save_res_path` → `<name>/output/` (not aug)
- `character_dict_path` → `<name>_aug/dict.txt` (copied by preprocess_recaug)
- NO `RecAug` in transforms (already pre-applied)
- `checkpoints` field must be empty/cleared before export
