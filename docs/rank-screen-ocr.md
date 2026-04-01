# Rank Screen OCR — Implementation Plan

Extract competitive rank progression data from the post-match rank screen.

## Target screen

The "COMPETITIVE VICTORY/DEFEAT" screen shown after ranked matches. Contains:

1. **Rank + division** — e.g. "GOLD 3", centered
2. **Rank progress (total)** — e.g. "RANK PROGRESS: 24%", below the rank emblem
3. **Progress bar with delta** — green (+N%) or red (-N%) segment, or "DEMOTION PROTECTION" overlay
4. **Modifiers** — tags below the progress bar

## What to extract

### 1. Rank + division

- **Font:** Big Noodle Titling Oblique
- **Position:** Fixed, center of screen, below the rank emblem
- **Values:** `BRONZE | SILVER | GOLD | PLATINUM | DIAMOND | MASTER | GRANDMASTER | CHAMPION` + division `1–5`
- **Approach:** Train a new PaddleOCR recognition model on Big Noodle Oblique with full uppercase alphabet + digits 1–5. The existing `team_side` model only knows ATTACK/DEFEND characters and can't be reused.

### 2. Rank progress (total %)

- **Font:** Futura (or similar geometric sans)
- **Position:** Fixed, left-aligned above the progress bar. Format: `RANK PROGRESS: N%`
- **Colors:** "RANK PROGRESS:" is white, the percentage is blue (positive) or orange (negative)
- **Approach:** Binarize the region by brightness (both blue and orange are bright on the dark background). OCR with existing `panel_values` model (Futura, digits + `%`) for the number. The label text is static and doesn't need OCR — just validate it's present.

### 3. Progress bar delta

Three visual states:

| State | Bar color | Text | Example |
|-------|-----------|------|---------|
| Positive | Green segment | `+N%` in white, `>` icons for each positive modifier | `+18% >>` |
| Negative | Red segment | `-N%` in white | `-27%` |
| Demotion protection | Full bar, distinct style | `DEMOTION PROTECTION` text | no delta available |

- **Approach:**
  - Detect bar color (green vs red vs demotion style) via HSV sampling
  - OCR the `+N%` / `-N%` text with `panel_values` model (needs `+` and `-` added, or just detect sign from bar color)
  - If demotion protection: flag it, delta is `None`
  - Count `>` / `<` chevron icons via blob detection or template match (optional, indicates modifier count)

### 4. Modifiers

Tags displayed below the progress bar in a horizontal row. Variable position.

**Known modifiers:**

| Modifier | Meaning |
|----------|---------|
| VICTORY | Won the match |
| DEFEAT | Lost the match |
| DRAW | Match ended in a draw |
| UPHILL BATTLE | Won despite being unfavored |
| EXPECTED | Won a match you were favored to win |
| CONSOLATION | Lost a match you were expected to lose (reduced penalty) |
| REVERSAL | Lost despite being favored |
| WINNING STREAK | On a win streak (may appear as WINNING TREND) |
| LOSING STREAK | On a loss streak (may appear as LOSING TREND) |
| CALIBRATION | Rank still being determined |
| VOLATILE | Lost after recently ranking up |
| DEMOTION PROTECTION | One more loss will demote |
| DEMOTION | Demoted after losing in demotion protection |
| WIDE | High rank disparity in lobby |
| PRESSURE | Pushes extreme-rank players toward average |

- **Font:** Futura (or similar)
- **Approach:** OCR the full region below the progress bar. Fuzzy-match extracted text against the known modifier list. Multiple modifiers can appear simultaneously (e.g. "DEFEAT", "DEMOTION", "LOSING TREND").

## Implementation phases

### Phase 1 — Standalone script + region mapping

Build `tools/rank_screen.py` that takes a frame image and extracts all 4 fields.

1. Load the 4 sample frames (frame_10s.png from each recording)
2. Manually identify pixel regions for each field at the recording resolution
3. Crop each region, binarize, visualize — confirm the regions are correct
4. Try existing OCR models (`panel_values` for digits, `panel_featured` for larger text) on the cropped regions
5. Evaluate what works and what needs new models

### Phase 2 — New OCR model for rank text

Train a PaddleOCR recognition model for Big Noodle Oblique:

- **Character set:** A–Z + 1–5 + space
- **Training data:** Synthetic renders of rank names + divisions at various sizes
- **Follow:** `docs/training-ocr-models.md` for the existing training pipeline

### Phase 3 — Full extraction pipeline

1. Rank + division OCR (new model)
2. Rank progress % OCR (existing `panel_values` or new model)
3. Progress bar state detection (HSV color) + delta OCR
4. Modifier OCR + fuzzy matching against known list
5. Return structured result: `{rank, division, progress_pct, delta_pct, demotion_protection, modifiers}`

### Phase 4 — Integration

Wire into the main app:

- Detect when the rank screen is visible (after `match_ended` Overwolf event)
- Run extraction on the frame
- Include rank progression data in the match summary and MCP payload
