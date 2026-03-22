# Hero Panel & Rank Detection

Technical reference for extracting hero stats and rank information from Overwatch 2 tab screenshots.

## Hero Panel Crop

The hero stats panel is a dark blue rectangle on the right side of the tab screen. It contains a featured stat, hero portrait + name, and a list of stat label/value pairs.

**Detection method:** Scan for long contiguous runs of the panel's dark blue color (~RGB 14,18,32 / ~RGB 29,37,58). The panel is semi-transparent, so exact colors vary.

1. Restrict search to right 45% of the image, skip top 12% (HUD)
2. Create a color mask: `R=5-50, G=10-55, B=25-80, B > R, B > G`
3. For each column, find the longest vertical run of matching pixels
4. For each row, find the longest horizontal run
5. Columns with vertical runs > 40% of ROI height are panel edges → leftmost = left edge, rightmost = right edge
6. Rows with horizontal runs > 50% of panel width → topmost = top edge, bottommost = bottom edge

**Typical panel size at 4K (3840×2160):** ~869×1531 pixels.

## Featured Stat

The featured stat sits in a darker rounded rectangle in the top-right of the panel (overlapping with the hero portrait area).

**Detection method:** Color-based contour detection in the top 25% of the panel, right third.

1. Mask for the darker box color: `R=5-30, G=10-35, B=20-50, B > R`
2. Morphological close + open to clean the mask
3. Find the largest contour → bounding rect = featured stat box

**Structure:** Big value on top (~60% of box height), label text below (~40%). Split vertically, OCR each half separately.

## Label-Value Separation

Stat rows below the hero portrait follow a repeating pattern: bold white value, then smaller gray label underneath.

**Detection method:** Horizontal brightness projection.

1. For each row of the panel, count pixels with brightness > 80
2. Rows where bright pixel percentage > 1% of width are text rows
3. Group consecutive text rows into blocks (merge gaps < 4px, discard blocks < 6px tall)
4. Classify each block:
   - Height > 100px → `header` (portrait + featured stat area)
   - Average brightness of text pixels > 170 → `value` (white text)
   - Otherwise → `label` (gray text)

**Strip cutting:** Labels define the strip boundaries. Each stat strip spans from the bottom of the previous label (+ 6px padding) to the bottom of the current label (+ 6px padding). This gives consistent ~118px strips, each containing one value and one label.

## OCR Models

Three separate PaddleOCR recognition models, finetuned from PP-OCRv5 server rec (PPHGNetV2_B4 backbone + SVTR neck + MultiHead CTC/NRTR):

### Labels Model (v1)

- **Font:** Config Medium ([globalfonts.pro/font/config](https://globalfonts.pro/font/config))
- **Character set:** A-Z + space (27 characters)
- **Training data:** 5,000 synthetic samples, white text on black background, font sizes 24-44px (weighted toward 34px for 4K)
- **Training config:** RecAug enabled, 100 epochs, LR 5e-5, batch 64
- **Accuracy:** 98.5% on synthetic val, **100% on real game screenshots** (19/19 across 3 test panels)
- **Inference:** Feed the full-width label row directly to the model (cropping to text bounds gives identical results but is unnecessary); `.strip()` the output to remove occasional leading spaces

### Values Model (v4)

- **Font:** Futura No2 Demi Bold
- **Character set:** 0-9 + % + , + . + : (14 characters)
- **Training data:** 5,000 synthetic samples, white text on black background, font sizes 32-60px (weighted toward 46px for 4K). Heavy oversampling of: zeros, standalone `1`, comma numbers, timer patterns
- **Training config:** RecAug enabled, 100 epochs, LR 5e-5, batch 64
- **Accuracy:** ~100% on synthetic val, **100% on real game screenshots** with proper cropping
- **Inference:** Pad value row ±20px vertically, then crop to text bounds (bright pixels > 120, 10px padding) before feeding to model. This is critical — without cropping, thin glyphs like `1` are lost when the 869px strip is resized to 320px

### Featured Model

- **Font:** Big Noodle Titling Oblique ([Resike/Overwatch](https://github.com/Resike/Overwatch/tree/master/Fonts))
- **Character set:** 0-9 + % + , + . + : (14 characters, same as values)
- **Training data:** 5,000 synthetic samples, white text on black background, font sizes 48-96px (larger than regular values, weighted toward 80px for 4K). Heavy oversampling of timer patterns (MM:SS) and 0-vs-7 confusable pairs
- **Training config:** RecAug pre-applied offline, 100 epochs, LR 5e-5, batch 64
- **Accuracy:** 99.8% on synthetic val, correct on real screenshots including timer values
- **Inference:** Same as values model — crop to text bounds before feeding to model

The featured stat uses a completely different font from the regular stat values (Big Noodle Titling vs Futura No2 Demi Bold). The Futura-trained values model misread `01:09` as `71:` because the italic Noodle `0` looks like a Futura `7`.

### Why Three Models

The hero panel uses three different fonts across four text elements:
- **Config Medium** — stat labels (CHARGE KILLS, FIRE STRIKE ACCURACY...)
- **Futura No2 Demi Bold** — stat values (5, 3, 29%...)
- **Big Noodle Titling** — featured stat value (01:09, 5, 29%...)
- **Unknown** — hero name (REINHARDT) — not OCR'd, comes from Overwolf roster

Splitting by font eliminates cross-domain confusion:
- No `0` vs `O` ambiguity (each model only knows one)
- No `1` vs `I` or `l` confusion
- No `0` vs `7` confusion from font style differences
- Restricted character sets converge faster and achieve higher accuracy

### Key Findings from Training

- **Clean white-on-black training data works better than varied colors.** A 69K model trained with color/background variations scored 99.5% on synthetic data but 0% on real game text — it overfit to PIL's font rendering. The 5K clean model generalizes perfectly.
- **RecAug (PaddleOCR's built-in augmentation) helps generalization** through spatial perturbations (perspective, crop, TIA distortion). However, running it during training is extremely slow on Windows — it runs single-threaded on CPU and the GPU sits idle waiting (`avg_reader_cost` jumps from 0.002s to 0.15s+ per batch, doubling total training time). The recommended approach is to pre-apply RecAug to the training images as a separate offline step (`tools/preprocess_recaug.py`), then train without RecAug on the pre-augmented data. This decouples CPU augmentation from GPU training.
- **Cropping to text bounds is essential for values.** PaddleOCR resizes all input to 48×320px. An 869px wide strip with a single `1` in the corner becomes invisible after resize.

## Rank Detection

Competitive matches display rank icons and division signs in the top-right corner of the tab screen.

**Method:** Multi-scale template matching with datamined game assets.

### Rank Icons

7 tiers: Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster, Champion.

Assets: `datamined_assets/{tier}_a.png` (128×128px native resolution)

1. Restrict search to top 20% of image, right 20% (the rank display area)
2. For each rank template, try scales 0.2-1.2 (step 0.05)
3. At each scale, run `cv2.matchTemplate` with `TM_CCOEFF_NORMED`
4. Collect all matches above 0.88 threshold
5. NMS: suppress matches within 50px of a higher-scoring match
6. Store **center x** of each match bounding box for position pairing
7. Sort by x position: leftmost = min rank, rightmost = max rank

**Typical match scale:** 0.75 at 4K, 0.40 at 1080p.

### Division Signs

5 divisions (1 = highest, 5 = lowest).

Assets: `datamined_assets/division_{1-5}_a.png` (128×60px native)

Same template matching process as rank icons. Division signs appear below their corresponding rank icon, horizontally centered.

**Pairing:** Each division is paired with the rank icon whose **center x** is closest. This handles both 4K and 1080p where the rank icons are at different spacings.

### Wide Match Detection

Some competitive matches are flagged as "wide match" (large skill gap between players). The game displays a yellow icon and "WIDE MATCH" text to the left of the rank icons.

**Method:** Check for yellow/gold pixels (the wide match icon color) in a 10%×10% region to the left of the leftmost rank icon.

1. Determine leftmost rank icon x position (~91.5% of image width at 4K)
2. Crop a region: x from `icon_x - 10%w` to `icon_x`, y from `5%h` to `15%h`
3. Count pixels matching yellow/gold: `R > 180, G in 140-220, B < 80`
4. If yellow pixel percentage > 0.5% → wide match

**Accuracy:** 26/26 on real competitive match screenshots.

### Rank Detection Performance

- **4K (3840×2160):** 26/26 correct (100%)
- **1080p (1920×1080):** 26/26 correct (100%)
- **882p (downscaled):** Rank tiers detected, divisions too small to match reliably
- **Speed:** ~1.2 seconds per screenshot (234 matchTemplate calls across 13 templates × 18 scales)
