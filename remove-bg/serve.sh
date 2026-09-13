#!/bin/sh
# Make sure server.py is awake on http://127.0.0.1:8777, then return: start it
# in the background if nothing listens, wake it if sleeper.py holds the port.
# Used by the Mac app and the Finder Quick Action (see make-app.sh), which
# have no terminal: errors go to a dialog, output to ~/Library/Logs/remove-bg.log.
cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:8777"
LOG="$HOME/Library/Logs/remove-bg.log"

status() { /usr/bin/curl -s -m 1 "$URL/api/status"; }     # empty when nothing listens
alert() {
  /usr/bin/osascript -e "display dialog \"$1\" with title \"Remove Background\" buttons {\"OK\"} with icon stop" >/dev/null
}

case "$(status)" in
  *loaded*) exit 0 ;;
  *sleeping*) /usr/bin/curl -s -m 2 -X POST -o /dev/null "$URL/api/wake" ;;
  *)
    if [ ! -x .venv/bin/python ]; then
      alert "Project not found in $(pwd). Run ./make-app.sh again from the remove-bg folder."
      exit 1
    fi
    nohup .venv/bin/python server.py --no-warmup > "$LOG" 2>&1 &
    ;;
esac

i=0
until status | grep -q loaded; do
  i=$((i + 1))
  if [ $i -gt 240 ]; then
    alert "The server did not start. Details: $LOG"
    exit 1
  fi
  sleep 0.5
done
