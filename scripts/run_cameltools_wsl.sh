#!/bin/bash
# -*- coding: utf-8 -*-
# Setup and run CAMeL Tools analysis in WSL

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "CAMeL Tools Setup & Analysis Pipeline (WSL)"
echo "========================================================================"
echo ""

# Check if in WSL
if ! grep -qi microsoft /proc/version; then
    echo "[WARNING] Not running in WSL. This script is optimized for WSL."
    echo "[INFO] You can still run it, but some features may not work."
    echo ""
fi

# ========================================================================
# Step 1: Create Python venv if needed
# ========================================================================

echo "[1] Checking Python environment..."

VENV_PATH="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "    Creating virtual environment..."
    python3 -m venv "$VENV_PATH" || {
        echo "[ERROR] Failed to create virtual environment"
        echo "Try: python3 -m venv $VENV_PATH"
        exit 1
    }
    echo "    ✓ Virtual environment created"
fi

# Verify activation script exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "[ERROR] Virtual environment activation script not found"
    echo "Path: $VENV_PATH/bin/activate"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"
echo "    ✓ Virtual environment activated"
echo "    Using Python: $(which python3)"

# ========================================================================
# Step 2: Install dependencies
# ========================================================================

echo ""
echo "[2] Installing/checking dependencies..."

# Check if camel-tools is installed
if ! python3 -c "import camel_tools" 2>/dev/null; then
    echo "    Installing camel-tools..."
    pip install -q camel-tools
    echo "    ✓ camel-tools installed"
else
    echo "    ✓ camel-tools already installed"
fi

# Verify other requirements
pip install -q -r requirements.txt 2>/dev/null || true

# ========================================================================
# Step 3: Run analysis
# ========================================================================

echo ""
echo "[3] Running analysis..."
echo ""

python3 "$PROJECT_ROOT/scripts/analyze_with_cameltools.py"

# ========================================================================
# Step 4: Report results
# ========================================================================

echo ""
echo "[4] Results saved to:"
echo "    - results/cameltools_pattern_comparison.txt"
echo "    - results/cameltools_isnad_analysis.json"
echo "    - results/cameltools_khabar_analysis.json"
echo ""
echo "========================================================================"
echo "Analysis complete!"
echo "========================================================================"
