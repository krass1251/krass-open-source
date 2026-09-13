#!/bin/sh
# Build "Remove Background.app" in ~/Applications, so the UI starts from
# Launchpad / Spotlight / the Dock without a terminal:
#   ./make-app.sh
# The app starts server.py in the background if it is not running yet, then
# opens http://127.0.0.1:8777 in Chrome (default browser if Chrome is missing).
# Stop the server with the Quit button on the page. Re-run after moving the repo.
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
APP="$HOME/Applications/Remove Background.app"

[ -x .venv/bin/python ] || ./setup.sh

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Remove Background</string>
  <key>CFBundleDisplayName</key><string>Remove Background</string>
  <key>CFBundleIdentifier</key><string>local.krass.remove-bg</string>
  <key>CFBundleExecutable</key><string>launch</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/launch" <<'SH'
#!/bin/sh
DIR="__DIR__"
URL="http://127.0.0.1:8777"
LOG="$HOME/Library/Logs/remove-bg.log"

up() { curl -s -o /dev/null -m 1 "$URL/api/status"; }
alert() { osascript -e "display dialog \"$1\" with title \"Remove Background\" buttons {\"OK\"} with icon stop"; }

if ! up; then
  if [ ! -x "$DIR/.venv/bin/python" ]; then
    alert "Project not found in $DIR. Run ./make-app.sh again from the remove-bg folder."
    exit 1
  fi
  cd "$DIR" || exit 1
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
fi

open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
SH
sed -i '' "s|__DIR__|$DIR|" "$APP/Contents/MacOS/launch"
chmod +x "$APP/Contents/MacOS/launch"

# icon: dark rounded square, checkerboard "removed" half, white subject
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
.venv/bin/python - "$ICONSET" <<'PY'
import sys
from PIL import Image, ImageDraw
S = 1024
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((100, 100, 924, 924), radius=190, fill=255)
art = Image.new("RGBA", (S, S), (31, 31, 35, 255))
d = ImageDraw.Draw(art)
cell = 64
for y in range(100, 924, cell):
    for x in range(512, 924, cell):
        if ((x - 512) // cell + (y - 100) // cell) % 2 == 0:
            d.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(70, 70, 78, 255))
d.ellipse((352, 250, 672, 570), fill=(245, 245, 240, 255))
d.pieslice((232, 600, 792, 1160), 180, 360, fill=(245, 245, 240, 255))
icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(art, (0, 0), mask)
for px in (16, 32, 128, 256, 512):
    icon.resize((px, px), Image.LANCZOS).save(f"{sys.argv[1]}/icon_{px}x{px}.png")
    icon.resize((px * 2, px * 2), Image.LANCZOS).save(f"{sys.argv[1]}/icon_{px}x{px}@2x.png")
PY
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"
touch "$APP"

echo "built: $APP"
echo "open it from Launchpad or Spotlight (Remove Background), drag it to the Dock to keep it there"
