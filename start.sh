#!/bin/sh
# Start the dashboard (macOS / Linux):  sh start.sh
cd "$(dirname "$0")"
python3 run.py check >/dev/null 2>&1 || python3 -m pip install -r requirements.txt
python3 run.py
