# Hero Panel OCR Training Report

## Task

We need to OCR two types of text from Overwatch 2's hero stats panel (the right-side panel visible when pressing Tab in-game):

1. **Stat values** — numbers like `47%`, `2,470`, `0`, `100%`, `1,789`
   - Font: **Futura** (regular weight)
   - Appearance: bold white text (~RGB 245,245,245) on dark blue background (~RGB 14,18,32)
   - Character set: `0-9`, `%`, `,`, `.` (13 characters)

2. **Stat labels** — ALL CAPS descriptions like `WEAPON ACCURACY`, `EARTHSHATTER DIRECT HITS`, `COALESCENCE KILLS`
   - Font: **Config Medium** (a geometric sans-serif)
   - Appearance: gray text (~RGB 102,109,129) on dark blue background (~RGB 14,18,32)
   - Character set: `A-Z` + space (27 characters)

The panel background is semi-transparent, so the actual pixel colors vary slightly depending on the game scene behind it. The text is always horizontal, always the same font, always ALL CAPS for labels.

The screenshots are captured at 4K (3840x2160) via Windows Graphics Capture + NVENC H.265. The hero panel is approximately 869x1531 pixels. Individual stat label rows are ~25px tall. Individual stat value rows are ~33px tall.

## Architecture

We're finetuning **PP-OCRv5 server rec** (PaddleOCR's latest recognition model) with architecture:
- Backbone: PPHGNetV2_B4
- Neck: SVTR (dims=120, depth=2)
- Head: MultiHead (CTCHead + NRTRHead with nrtr_dim=384)
- Pretrained from: `PP-OCRv5_server_rec_pretrained.pdparams`

Two separate models are trained — one for labels (27 chars), one for values (13 chars). The restricted character sets eliminate cross-domain confusion (e.g., `0` vs `O` can't happen when the model only knows one of them).

## Inference Pipeline

The panel is detected from the full screenshot using color-based edge detection (scanning for long runs of the panel's dark blue color). Individual stat rows are found via horizontal brightness projection. Each row is cut into a value strip and a label strip. The strips are fed to the respective models.

The label strips from the real game are approximately 869px wide × 25px tall, with text occupying only the left ~40-60% of the width. The value strips are similar width × 33px tall, with text in the left ~10-30%.

PaddleOCR's `RecResizeImg` resizes all input to `[3, 48, 320]` (height 48, width 320) while preserving aspect ratio (padding with black on the right). This means a 869×25 strip gets scaled to roughly 48×600+, then the width is either truncated or padded to 320.

## Labels Model v1 (5K, clean white-on-black)

### Training Data Generation

- **Count**: 5,000 samples (4,500 train / 500 val)
- **Font**: Config-Medium.ttf at sizes [24, 26, 28, 30, 32, **34**, **34**, **34**, 36, 38, 40, 44] (weighted toward 34, which matches 4K panel)
- **Colors**: Pure white text (255,255,255) on pure black background (0,0,0)
- **No noise, no blur, no color variation**
- **Content mix**:
  - 40% known OW2 stat labels (e.g., "WEAPON ACCURACY", "EARTHSHATTER KILLS")
  - 30% random combinations of 1-5 known OW2 words
  - 15% random 1-5 gibberish ALL CAPS words (3-12 chars each)
  - 15% single long random word (8-20 chars)
- **Image dimensions**: tightly cropped to text with 4-20px horizontal padding, 4-12px vertical padding. Typical size ~200×40px for a 2-word label.

### Training Config

```yaml
epoch_num: 100
learning_rate: 0.00005
warmup_epoch: 1
L2_factor: 3.0e-05
batch_size: 64
num_workers: 4  # then changed to 0 due to Windows deadlock issues
max_text_length: 40
use_space_char: true
RecAug: enabled (default PaddleOCR augmentation)
image_shape: [3, 48, 320]
```

RecAug applies: TIA distortion/stretch/perspective (40% prob), random crop, Gaussian blur, HSV jitter, color jitter, Gaussian noise, and color inversion (40% prob).

### Training Results

- Epoch 7: acc 0.87
- Epoch 15: acc 0.96, norm_edit_dis 0.998
- Epoch 29: acc 0.97
- Epoch 72: acc 0.982
- **Best (epoch ~90): acc 0.985, norm_edit_dis 0.999**

### Real-World Test Results: **19/19 (100%)**

Tested on label rows extracted from 3 real game screenshots (Juno, Reinhardt, Moira panels):

```
WEAPON ACCURACY              → WEAPON ACCURACY              OK
PULSAR TORPEDOES DAMAGE      → PULSAR TORPEDOES DAMAGE      OK
PULSAR TORPEDOES HEALING     → PULSAR TORPEDOES HEALING     OK
ORBITAL RAY HEALING          → ORBITAL RAY HEALING          OK
DAMAGE AMPLIFIED             → DAMAGE AMPLIFIED             OK
ORBITAL RAY ASSIST           → ORBITAL RAY ASSIST           OK
CHARGE KILLS                 → CHARGE KILLS                 OK
FIRE STRIKE KILLS            → FIRE STRIKE KILLS            OK
FIRE STRIKE ACCURACY         → FIRE STRIKE ACCURACY         OK
KNOCKBACK KILLS              → KNOCKBACK KILLS              OK
EARTHSHATTER DIRECT HITS     → EARTHSHATTER DIRECT HITS     OK
EARTHSHATTER STUNS           → EARTHSHATTER STUNS           OK
EARTHSHATTER KILLS           → EARTHSHATTER KILLS           OK
SECONDARY FIRE ACCURACY      → SECONDARY FIRE ACCURACY      OK
BIOTIC ORB KILLS             → BIOTIC ORB KILLS             OK
BIOTIC ORB HEALING           → BIOTIC ORB HEALING           OK
ALLY COALESCENCE EFFICIENCY  → ALLY COALESCENCE EFFICIENCY  OK
ENEMY COALESCENCE EFFICIENCY → ENEMY COALESCENCE EFFICIENCY OK
COALESCENCE KILLS            → COALESCENCE KILLS            OK
```

Confidence scores ranged 0.45-0.79 (low but text was correct). Some results had leading spaces that needed `.strip()`.

Note: these real label strips were fed to the model **without any preprocessing** — raw BGR crops from the panel, 869px wide with text in the left portion, gray text on dark blue background. The model generalized from pure white-on-black training data to gray-on-dark-blue real data.

## Labels Model v2 (69K, color variations)

### Motivation for Changes

Based on PaddleOCR finetuning best practices research:
- Recommended 50K+ samples when changing character dictionary
- Recommended adding color/background variation for robustness
- Recommended stronger L2 regularization (1e-4 vs 3e-5)
- Recommended longer warmup (2 epochs vs 1)
- Recommended charset coverage samples (individual chars, pairs)

### Training Data Generation

- **Count**: 69,178 samples (~62K train / ~7K val) — 69,000 random + 178 charset coverage
- **Font**: Same Config-Medium.ttf, same size distribution
- **Colors**: VARIED (the key difference from v1)
  - Background colors: pure black (weighted 3x), actual panel dark blue (14,18,32), variations (20,25,40), (10,12,22), (25,30,45), neutral dark grays — all with ±10 RGB jitter
  - Text colors: pure white (weighted 3x), actual label gray (102,109,129), variations (110,115,135), (90,97,117), (120,125,140), light grays — all with ±10 RGB jitter
- **Noise**: 30% chance of Gaussian noise (intensity 0.02-0.08)
- **Blur**: 15% chance of Gaussian blur (radius 0.3-0.7)
- **Charset coverage**: every individual character rendered at 3 sizes, 100 random character pairs
- **Content mix**: same as v1 (40% real labels, 30% word combos, 15% gibberish, 15% long words)

### Training Config Diff from v1

```
warmup_epoch: 1 → 2
L2_factor: 3.0e-05 → 0.0001
RecAug: REMOVED (data already has built-in augmentation)
num_workers: 4 (kept)
```

RecAug was removed because it was causing extreme CPU bottleneck — `avg_reader_cost` went from 0.003s (without RecAug) to 0.15-0.34s (with RecAug), making training 2-3x slower with GPU at only 50% utilization. The synthetic data generator already provides color/noise/blur variation, so RecAug was double-augmenting.

### Training Results

- Epoch 2: acc 0.921
- Epoch 5: acc 0.970
- Epoch 20: acc 0.989
- **Best (epoch 86): acc 0.9951, norm_edit_dis 0.9997**

Higher synthetic accuracy than v1 (99.5% vs 98.5%).

### Real-World Test Results: **0/19 (0%)**

Complete failure on real game screenshots:

```
WEAPON ACCURACY              → EAPNIACRACY                  XX
PULSAR TORPEDOES DAMAGE      → PJLSARTIRPEDESDAMIAE         XX
EARTHSHATTER DIRECT HITS     → EARIHHIAITIERDIRETIHIITS     XX
SECONDARY FIRE ACCURACY      → ENDARYFIREACJRAY             XX
COALESCENCE KILLS            → CALESENIEKILLS               XX
```

### Diagnosis

The v2 model works perfectly on its own synthetic training data (9/10 correct on random training samples). It also reads PIL-rendered Config-Medium text perfectly. But it cannot read the actual game screenshots at all.

Key finding: the v1 model (trained only on white-on-black) reads the same real game strips at 100% accuracy. When tested on the same binarized+cropped image:
- v1: `[0.979] SECONDARY FIRE ACCURACY` ✓
- v2: `[0.908] ENDARYFIREACCRAY` ✗

The v2 model also fails on binarized (white-on-black) versions of real strips, even though it succeeds on PIL-rendered white-on-black text. The difference is subtle font rendering: PIL's text renderer produces slightly different anti-aliasing, kerning, and stroke characteristics than the game engine's renderer.

**Hypothesis**: The v2 model with 69K varied-color samples overfit to PIL's specific rendering characteristics. The color variations taught the model to rely on color-channel patterns rather than pure letterform shapes. The v1 model with 5K clean white-on-black samples learned more robust letterform features because there were no color patterns to latch onto.

## Values Model v2 (69K)

### Training Data Generation

- **Count**: 69,000+ samples
- **Font**: Futura.ttf at sizes [32, 34, 36, 38, 40, 42, 44, **46**, **46**, **46**, 48, 50, 52, 56, 60] (weighted toward 46, which matches 4K panel)
- **Colors**: Same varied scheme as labels v2 (backgrounds + text colors + jitter)
- **Noise/blur**: Same as labels v2 (30% noise, 15% blur)
- **Content mix**:
  - 20% zero ("0")
  - 10% small numbers (1-30)
  - 15% percentages (0-100%)
  - 15% comma numbers (1,000-99,999)
  - 10% three-digit numbers
  - 8% decimal percentages
  - 7% large comma numbers (100K+)
  - 5% single digits
  - 5% two-digit numbers
  - 5% random digit strings
- **Charset coverage**: each individual character at 4 sizes, confusable sequences (0, 00, 0%, 1, 1%, etc.)
- **Character dict**: `0123456789%,.` (13 chars)

### Training Config

```yaml
epoch_num: 100
learning_rate: 0.00005
warmup_epoch: 2
L2_factor: 0.0001
max_text_length: 15
use_space_char: false
RecAug: REMOVED
num_workers: 4
```

### Training Results

- **Best (epoch 90): acc 0.99999999, norm_edit_dis 1.0**

Essentially perfect on synthetic data.

### Real-World Test Results: **15/19 (78.9%)**

```
32%    → 32%      OK     47%   → 47%     OK
2,470  → 2.470    XX     0     → 0       OK
1,789  → 1.789    XX     2,035 → 2.035   XX
425    → 425      OK     100%  → 100%    OK
64     → 64       OK     45%   → 45%     OK
1      → (empty)  XX     0     → 0       OK
0      → 0        OK (×7)
50%    → 50%      OK
```

Failure patterns:
- **Comma vs period**: `2,470` → `2.470`, `1,789` → `1.789`, `2,035` → `2.035`. The model reads commas as periods consistently.
- **Thin "1"**: The digit `1` alone as a stat value is not detected at all (empty output).

## Comparative Summary

| Model | Train Data | Synthetic Acc | Real-World Acc | Notes |
|-------|-----------|---------------|----------------|-------|
| Labels v1 | 5K clean white-on-black | 98.5% | **100% (19/19)** | Winner for labels |
| Labels v2 | 69K varied colors | 99.5% | **0% (0/19)** | Complete failure on real data |
| Values v2 | 69K varied colors | 100% | **78.9% (15/19)** | Comma→period confusion, thin 1 missed |

## Key Questions

1. Why does the labels v1 model (5K, clean) generalize perfectly to real game text while v2 (69K, varied) fails completely? Both use the same architecture, same font, same character set. The only difference is v1 trained on white-on-black while v2 trained on varied colors.

2. For the values model, how can we fix the comma vs period confusion? Both `,` and `.` are in the character set. The training data has plenty of comma numbers. Yet the model consistently reads commas as periods on real game text.

3. What would be the optimal training data strategy for v3? Should we go back to clean white-on-black for values too? Or is there a middle ground that preserves robustness without the overfitting we saw in v2?

4. The v1 labels model was trained WITH RecAug (which includes color inversion, distortion, blur, noise). The v2 model was trained WITHOUT RecAug but with pre-baked color variation. Could RecAug's specific augmentation pipeline actually help generalization in a way that pre-baked variation doesn't?

5. Is the aspect ratio mismatch a factor? Training images are ~200×40px (compact) but real strips are 869×25px (very wide, text left-aligned). PaddleOCR resizes to [48, 320]. Could the v1 model be handling this resize better due to simpler training distribution?
