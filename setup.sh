#!/usr/bin/env bash
set -e
echo "========================================"
echo "  PSO Benchmark Suite — Setup"
echo "========================================"
if ! command -v python3 &> /dev/null; then
    echo "[!] Installing Python3..."; sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi
echo "[✓] $(python3 --version)"
if [ ! -d ".venv" ]; then
    echo "[*] Creating .venv ..."; python3 -m venv .venv; echo "[✓] venv created"
else echo "[✓] .venv exists"; fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q
mkdir -p results/{single,benchmarks,grid_search,viz,analysis,color_case,examples}
echo "[*] Running tests..."
python -m pytest tests/ -v --tb=short
echo "========================================"
echo "  Setup complete! Run: source .venv/bin/activate"
echo "========================================"
