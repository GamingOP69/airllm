@echo off
echo "=== AIRLLM ENVIRONMENT SETUP ==="

if exist .venv (
    echo "--- 1. Virtual Environment already exists, skipping creation. ---"
) else (
    echo "--- 1. Creating Virtual Environment... ---"
    python -m venv .venv
)

echo "--- 2. Upgrading Pip... ---"
.venv\Scripts\python.exe -m pip install --upgrade pip

echo "--- 3. Installing Dependencies... ---"
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo "=== SETUP COMPLETE! ==="
echo "To run AirLLM, use: run_gpt_oss.bat"
pause
