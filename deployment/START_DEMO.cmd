@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Cai moi truong Python 3.13. Can Internet cho lan cai dau.
  py -3.13 -m venv .venv
  if errorlevel 1 (
    py -3.12 -m venv .venv
    if errorlevel 1 goto fail
  )
)
".venv\Scripts\python.exe" -c "import torch,transformers,safetensors,numpy" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" web_demo.py --model GoEmotions_BERT.goemotions --open
if errorlevel 1 goto fail
exit /b 0
:fail
echo Khong khoi dong duoc. Can Python 3.12 hoac 3.13 va mang khi cai thu vien lan dau.
pause
exit /b 1
