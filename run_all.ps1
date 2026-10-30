# One-click pipeline for Windows PowerShell.
# Usage (from the project folder):  .\run_all.ps1
# Optional fast mode:               .\run_all.ps1 -Fast
param([switch]$Fast)

$ErrorActionPreference = "Stop"

if (-not (Test-Path venv)) {
    Write-Host "Creating virtual environment (Python 3.11)..."
    py -3.11 -m venv venv
}
. .\venv\Scripts\Activate.ps1

Write-Host "Installing requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Getting dataset..."
python download_dataset.py

if (-not (Test-Path models\traffic_sign_cnn.h5)) {
    Write-Host "Training model..."
    if ($Fast) { python train.py --max-per-class 300 --epochs 8 } else { python train.py }
} else {
    Write-Host "Model already trained, skipping training."
}

Write-Host "Starting web app..."
streamlit run app.py
