# Training OCR Models

Guide for training or retraining the hero panel OCR models from scratch on a freshly cloned repository.

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA support (tested on RTX 4080)
- PaddlePaddle GPU and PaddleOCR installed (`uv sync` handles this)
- The fonts used by Overwatch 2's hero panel (not included in the repo — these are paid fonts, must be sourced by the user):
  - **Config Medium** (`Config-Medium.ttf`) — for stat labels. Available at [globalfonts.pro/font/config](https://globalfonts.pro/font/config).
  - **Futura No2 Demi Bold** (`Futura No2 Demi Bold.ttf` or similar) — for stat values. A community-sourced version is available at [Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts) (listed as `Futura No 2 D DemiBold.ttf`).

## Repository Structure

```
tools/
  generate_label_training.py    # Synthetic data generator for labels
  generate_value_training.py    # Synthetic data generator for values
  preprocess_recaug.py          # Offline RecAug augmentation (optional)
training_data/
  PP-OCRv5_server_rec_pretrained.pdparams   # Base pretrained model (required)
  panel_labels/                 # Labels model training data + output
  panel_values_v4/              # Values model training data + output
paddleocr_repo/                 # PaddleOCR clone (used for training scripts)
overwatchlooker/models/         # Production inference models (git-tracked)
  panel_labels/                 # Exported labels model
  panel_values/                 # Exported values model
  rank_assets/                  # Rank icon templates (not trained, just assets)
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

Replace `<model>` with `panel_labels` or `panel_values` (or whatever directory name you used in Step 1).

### RecAug: Slow But Necessary

RecAug runs CPU-side augmentation (perspective warp, blur, noise, color inversion) during training. On Windows with `num_workers: 4` it roughly doubles training time. **Do not remove it** — our v1 labels model trained with RecAug achieves 100% on real data; v2 trained without it (but with pre-baked color variation) achieved 0%.

If training is too slow, you can pre-apply RecAug offline:

```bash
uv run python tools/preprocess_recaug.py \
  --input training_data/panel_labels \
  --output training_data/panel_labels_augmented \
  --variants 1 --workers 4
```

Then point the training config's `data_dir` at the augmented directory and remove `RecAug` from the transforms list. This decouples CPU augmentation from GPU training.

## Step 3: Train

```bash
uv run python -m paddleocr_repo.tools.train -c training_data/panel_values_v4_finetune.yml
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
  -c training_data/panel_values_v4_finetune.yml \
  -o "Global.pretrained_model=./training_data/panel_values_v4/output/best_accuracy" \
     "Global.save_inference_dir=./training_data/panel_values_v4/inference/"
```

**Important**: Remove or clear the `checkpoints` field in the config before exporting, otherwise export will fail or load the wrong weights.

The exported model has a `model_name` field in `inference.yml` that must match `PP-OCRv5_server_rec` for `paddlex.create_model` to load it:

```bash
# Fix model name in exported config
sed -i 's/panel_values_v4_rec/PP-OCRv5_server_rec/' training_data/panel_values_v4/inference/inference.yml
```

## Step 5: Test on Real Screenshots

Run the hero panel OCR script on test screenshots:

```bash
uv run python debug_panel_structure.py path/to/tab_screenshot.png
```

Use any Overwatch 2 tab screen screenshot with a hero stats panel visible. Expected output: every label and value read correctly. If not, check:

1. **Values read as empty or garbled**: Are you cropping to text bounds before feeding to the model? Full 869px strips must be cropped to just the text region (see `crop_to_text` in the script).
2. **Labels misread**: Check that the model was trained with RecAug enabled.
3. **Commas read as periods**: Ensure `,` is in the character dictionary and training data has comma-number samples.

## Step 6: Deploy

Copy the exported inference model to the production location:

```bash
cp training_data/panel_values_v4/inference/* overwatchlooker/models/panel_values/
cp training_data/panel_values_v4/dict.txt overwatchlooker/models/panel_values/
```

Commit via git (`.pdiparams` files are tracked via Git LFS).

## Lessons Learned

These are hard-won findings from multiple training iterations. Read before changing anything.

### What Works

- **5K clean white-on-black samples** — sufficient for 100% real-world accuracy with restricted character sets
- **RecAug during training** — provides spatial robustness (perspective, crop, distortion) that bridges the synthetic-to-real gap
- **Restricted character sets** — A-Z+space for labels, 0-9+%+,+.+: for values. Eliminates O/0 and I/1 confusion entirely
- **Separate models for labels and values** — different fonts (Config Medium vs Futura), different character sets, cleaner results

### What Does NOT Work

- **Color/background variation in training data** — 69K samples with varied colors scored 99.5% on synthetic data but **0% on real game screenshots**. The model overfits to PIL's font rendering artifacts instead of learning robust letterforms.
- **Removing RecAug** — without spatial augmentation the model doesn't generalize from synthetic to real, even with more training data
- **Large training sets without RecAug** — more data doesn't help if the model can't bridge the rendering gap
- **Feeding full-width strips to the values model** — 869px strips with a single thin character (like `1`) get destroyed when resized to 320px. Always crop to text bounds first.

### Architecture Reference

Both models use the same architecture (finetuned from PP-OCRv5 server rec):

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
