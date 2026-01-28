@echo off
if not exist .venv\Scripts\python.exe (
    echo "=== ERROR: Virtual environment not found. Please run setup.bat first. ==="
    pause
    exit /b
)
echo "=== RUNNING AIRLLM: GPT-OSS-20B ==="
.venv\Scripts\python.exe scripts\run_gpt_oss.py
pause
