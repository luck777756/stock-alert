#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt

exec python3 main.py
