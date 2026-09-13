#!/bin/sh
# Make sure server.py is running in the background, then return. Used by the
# Mac app and the Finder Quick Action (see make-app.sh), which have no
# terminal: errors go to a dialog, output to ~/Library/Logs/remove-bg.log.
cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:8777"
LOG="$HOME/Library/Logs/remove-bg.log"

up() { /usr/bin/curl -s -o /dev/null -m 1 "$URL/api/status"; }
alert() {
  /usr/bin/osascript -e "display dialog \"$1\" with title \"Remove Background\" buttons {\"OK\"} with icon stop" >/dev/null
}

up && exit 0
if [ ! -x .venv/bin/python ]; then
  alert "Project not found in $(pwd). Run ./make-app.sh again from the remove-bg folder."
  exit 1
fi
nohup .venv/bin/python server.py --no-warmup > "$LOG" 2>&1 &
i=0
until up; do
  i=$((i + 1))
  if [ $i -gt 240 ]; then
    alert "The server did not start. Details: $LOG"
    exit 1
  fi
  sleep 0.5
done
