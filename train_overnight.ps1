$ErrorActionPreference = "Continue"
# Values v3 training: generate clean data, pre-augment with RecAug, train.
# Usage: .\train_overnight.ps1

$logFile = "training_data\train_values_v3_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"

Write-Host "=== Values v3 training started at $(Get-Date) ===" -ForegroundColor Cyan
Write-Host "Log file: $logFile"

Write-Host "`n=== Step 1: Generate 69K clean value samples ===" -ForegroundColor Cyan
cmd /c "uv run python tools/generate_value_training.py --count 50000 --output training_data/panel_values_v3 2>&1" | Tee-Object -FilePath $logFile

Write-Host "`n=== Step 2: Pre-augment with RecAug (3 variants per image) ===" -ForegroundColor Cyan
cmd /c "uv run python tools/preprocess_recaug.py --input training_data/panel_values_v3 --output training_data/panel_values_v3/augmented --variants 1 --workers 4 2>&1" | Tee-Object -Append -FilePath $logFile

Write-Host "`n=== Step 3: Train values model (100 epochs) ===" -ForegroundColor Cyan
cmd /c "uv run python -m paddleocr_repo.tools.train -c training_data/panel_values_v3_finetune.yml 2>&1" | Tee-Object -Append -FilePath $logFile

Write-Host "`n=== Done at $(Get-Date) ===" -ForegroundColor Green
Write-Host "Best model: training_data/panel_values_v3/output/best_accuracy"
Write-Host "Log: $logFile"
