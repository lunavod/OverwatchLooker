$ErrorActionPreference = "Stop"
# One-time setup for the PaddleOCR training & export environment.
# Run from everything_paddle/:  .\init.ps1

Write-Host "=== Step 1/3: Creating Python 3.11 environment ===" -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

Write-Host "`n=== Step 2/3: Cloning PaddleOCR (v3.1.1) ===" -ForegroundColor Cyan
if (Test-Path paddleocr_repo) {
    Write-Host "  paddleocr_repo/ already exists, skipping clone"
} else {
    git clone --branch v3.1.1 --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git paddleocr_repo
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

Write-Host "`n=== Step 3/3: Downloading pretrained model ===" -ForegroundColor Cyan
$weightsFile = "training_data\PP-OCRv5_server_rec_pretrained.pdparams"
if (Test-Path $weightsFile) {
    Write-Host "  Pretrained weights already exist, skipping download"
} else {
    Write-Host "  Download the pretrained weights manually:"
    Write-Host "  https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec" -ForegroundColor Yellow
    Write-Host "  Place at: $weightsFile" -ForegroundColor Yellow
}

Write-Host "`n=== Verifying installation ===" -ForegroundColor Cyan
uv run python -c "import paddle; print(f'PaddlePaddle {paddle.__version__}  GPU: {paddle.device.is_compiled_with_cuda()}')"
uv run python -c "import paddle2onnx; print(f'paddle2onnx {paddle2onnx.__version__}')"

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "See docs/training-ocr-models.md for usage."
