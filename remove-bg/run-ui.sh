#!/bin/sh
# Start the local drag-and-drop UI on http://127.0.0.1:8777
# Runs setup.sh first if the .venv is missing (fresh clone).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python server.py "$@"
