#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Py3D Modeler: Building Single-File Standalone Binary ==="

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pyinstaller
.venv/bin/pyinstaller --clean --noconfirm Py3DModeler.spec

echo ""
echo "SUCCESS! Standalone binary built at: $(pwd)/dist/Py3DModeler"
