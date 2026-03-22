# Score Detection

Technical reference for detecting round scores from the Overwatch 2 in-game HUD during Control and Flashpoint matches.

## Overview

Control and Flashpoint modes display score indicators in the top-center HUD — small shapes (circles for Control, diamonds for Flashpoint) that fill with color when a team wins a round point. The system detects these by sampling specific pixel positions across multiple frames and checking both color and temporal stability.

**Supported modes:**

| Mode | Indicators | Max Score | Shape |
|------|-----------|-----------|-------|
| Control | 2 per side | 2 | Circles |
| Flashpoint | 3 per side | 3 | Diamonds (rotated squares) |

Only active in quickplay (not ranked).

## Detection Method

### Fixed-Position Pixel Sampling

Unlike contour-based or template-matching approaches, the system samples a 5×5 pixel patch at each **known indicator center position**. These positions are fixed fractions of the screen size, so they scale with resolution.

This approach is immune to background color interference — a problem that made color-mask and contour-based approaches unreliable on maps with colored environments (e.g. Aatlis has extensive blue/red architectural elements that pass HSV thresholds).

### Indicator Positions

All positions are fractions of screen width (X) and height (Y), calibrated at 1080p.

**Flashpoint (3 diamonds per side):**

| Indicator | Blue X | Red X |
|-----------|--------|-------|
| 1st (outer) | 0.3594 | 0.5969 |
| 2nd (middle) | 0.3812 | 0.6188 |
| 3rd (inner) | 0.4031 | 0.6406 |

Y positions: `0.1069` (with banner) / `0.0699` (without banner)

**Control (2 circles per side):**

| Indicator | Blue X | Red X |
|-----------|--------|-------|
| 1st (outer) | 0.384 | 0.587 |
| 2nd (inner) | 0.413 | 0.616 |

Y positions: `0.0769`

### Banner Shift

The game sometimes displays a text banner above the score indicators ("FLASHPOINT UNLOCKS IN", "PREPARE TO ATTACK", etc.). When the banner is visible, the score UI is pushed down. The system checks both Y positions and accepts whichever has a valid reading.

### Temporal Variance Filtering

The key insight: filled HUD indicators are **opaque at their center**. Even though the indicators have slight transparency at the edges, the center pixels render identically every frame. Background elements — even stable ones like distant buildings or sky — have non-zero variance due to camera movement, compression artifacts, and lighting changes.

**Algorithm:**

1. Buffer the last 10 full frames
2. For each indicator position, extract the 5×5 patch from all 10 frames
3. Compute per-pixel standard deviation across the 10 frames (averaged across RGB channels)
4. If std > 5.0 → background pixel, skip
5. If std ≤ 5.0 → check HSV color of the middle frame's patch

In practice, filled indicator centers have std ≈ 0.0, while even stable-looking backgrounds have std > 20.

### Color Thresholds

Once temporal stability is confirmed, the patch is converted to HSV and checked:

**Blue (ally) filled indicator:**
- Hue: 80–120
- Saturation: > 170
- Value: > 150

**Red (enemy) filled indicator:**
- Hue: > 140 OR < 15 (wraps around 0/180 in OpenCV HSV)
- Saturation: > 150
- Value: > 150

Empty indicators (outlines) have lower saturation at their center and are not detected — they don't need to be, since we only count filled ones.

## Score Tracking

### Monotonic Increase

Score can only go up. If a reading shows a lower score than the last confirmed reading, it's rejected. This prevents transient false reads from corrupting the score history.

### Death Camera Invalidation

When the player dies, the game shows a "kill cam" replay from the killer's perspective. This can display the HUD at different positions or with different content. All readings within a window around each death event are rejected:

- **5 seconds before** the death event (the transition to kill cam starts slightly before the death event fires)
- **15 seconds after** the death event

### Round Gate

Detection doesn't start until the first `RoundStartEvent` from Overwolf. This avoids false reads during loading screens and hero select.

### Match-End Inference

If a match ends (VICTORY/DEFEAT) but neither team has reached the max score in the detected progression, the final score is inferred:
- VICTORY → blue team gets max score
- DEFEAT → red team gets max score

This handles cases where the player dies right before the winning point or has the tab screen open during the transition.

## Output

Score transitions are stored as a list of `(blue, red)` tuples starting from `(0, 0)`:

```
Score: 0:0 -> 1:0 -> 1:1 -> 2:1
```

Sent to MCP as `score_progression` (excluding the initial 0:0):

```json
["1:0", "1:1", "2:1"]
```

## Previous Approaches (What Didn't Work)

### Color Mask + Contour Detection

Initial approach: create HSV color masks for blue/red, find contours, check circularity and fill ratio (filled ~1.0 vs outline ~0.3).

**Problem:** Maps like Aatlis have blue/red architectural elements in the background that create contours matching the size, shape, and color of score indicators. No amount of contour filtering eliminates them.

### Temporal Color Presence

Count how many of 10 buffered frames have each pixel as blue/red. Keep pixels present in ≥7 frames.

**Problem:** Distant stable background elements (buildings, sky) persist across frames even with camera movement. A blue building at the indicator position passes the presence check.

### Per-Pixel Variance (Full ROI)

Compute per-pixel std across frames, keep only stable pixels (std < 3).

**Problem:** HUD indicators are semi-transparent — their rendered color changes slightly depending on the background behind them. When the camera moves, the background under the indicators shifts, causing the indicator pixels to have non-zero variance even though the HUD element itself is fixed. This filtered out HUD pixels along with background.

### Solution: Center-Pixel Sampling

The breakthrough: at the **exact center** of a filled indicator, the pixel is fully opaque (not semi-transparent). This means std ≈ 0.0 regardless of what's behind it. By sampling only at known center positions instead of analyzing the full ROI, we get:

- **Zero false positives from background** — background pixels always have std > 5
- **No shape/contour analysis needed** — we know exactly where to look
- **No color mask noise** — only 6-8 specific pixels are checked
- **Resolution independent** — positions are screen fractions
