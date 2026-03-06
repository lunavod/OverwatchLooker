"""Test script: run subtitle OCR on a screenshot and print results."""

import sys

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_VALUE_MIN = 180
_SAT_MAX_WHITE = 30
_SAT_MIN_COLOR = 80

# Wider region to avoid clipping text
_REGION_Y_START = 0.88
_REGION_Y_END = 1.00
_REGION_X_START = 0.20
_REGION_X_END = 0.80

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <screenshot_path>")
    sys.exit(1)
path = sys.argv[1]

img = cv2.imread(path)
h, w = img.shape[:2]

# Crop subtitle region
x1, x2 = int(w * _REGION_X_START), int(w * _REGION_X_END)
y1, y2 = int(h * _REGION_Y_START), int(h * _REGION_Y_END)
crop = img[y1:y2, x1:x2]

print(f"Image: {w}x{h}")
print(f"Subtitle region: ({x1},{y1}) -> ({x2},{y2}) = {x2-x1}x{y2-y1}")

# Build text mask
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
bright = hsv[:, :, 2] >= _VALUE_MIN
white_mask = bright & (hsv[:, :, 1] <= _SAT_MAX_WHITE)
color_mask = bright & (hsv[:, :, 1] >= _SAT_MIN_COLOR)
text_mask = white_mask | color_mask

print(f"White pixels: {np.count_nonzero(white_mask)}")
print(f"Color pixels: {np.count_nonzero(color_mask)}")
print(f"Total text pixels: {np.count_nonzero(text_mask)}")

# Binarize
binary = np.zeros(crop.shape[:2], dtype=np.uint8)
binary[text_mask] = 255

# Dilate slightly to connect broken character strokes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary = cv2.dilate(binary, kernel, iterations=1)

# Scale up 3x for better accuracy
binary_3x = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
# Re-threshold after cubic interpolation to keep it crisp
_, binary_3x = cv2.threshold(binary_3x, 128, 255, cv2.THRESH_BINARY)

# Save debug images
cv2.imwrite("debug_subtitle_crop.png", crop)
cv2.imwrite("debug_subtitle_binary.png", binary)
cv2.imwrite("debug_subtitle_binary_3x.png", binary_3x)
print("\nSaved: debug_subtitle_crop.png, debug_subtitle_binary.png, debug_subtitle_binary_3x.png")

# Run Tesseract
for psm in [6, 11]:
    text = pytesseract.image_to_string(binary_3x, config=f"--psm {psm}").strip()
    print(f"\n[PSM {psm}] {text!r}")

# Also try raw crop for comparison
raw_text = pytesseract.image_to_string(crop, config="--psm 6").strip()
print(f"\n[RAW PSM 6] {raw_text!r}")
