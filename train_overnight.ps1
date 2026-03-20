$ErrorActionPreference = "Continue"
# Overnight training: train both models sequentially.
# Data should already be generated in panel_labels_v2 and panel_values_v2.
# Usage: .\train_overnight.ps1

$logFile = "training_data\train_overnight_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"

Write-Host "=== Overnight training started at $(Get-Date) ===" -ForegroundColor Cyan
Write-Host "Log file: $logFile"

Write-Host "`n=== Step 1: Train labels model (100 epochs) ===" -ForegroundColor Cyan
cmd /c "uv run python -m paddleocr_repo.tools.train -c training_data/panel_labels_v2_finetune.yml 2>&1" | Tee-Object -FilePath $logFile

Write-Host "`n=== Step 2: Train values model (100 epochs) ===" -ForegroundColor Cyan
cmd /c "uv run python -m paddleocr_repo.tools.train -c training_data/panel_values_v2_finetune.yml 2>&1" | Tee-Object -Append -FilePath $logFile

Write-Host "`n=== Done at $(Get-Date) ===" -ForegroundColor Green
Write-Host "Labels best: training_data/panel_labels_v2/output/best_accuracy"
Write-Host "Values best: training_data/panel_values_v2/output/best_accuracy"
