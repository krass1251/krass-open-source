#!/bin/sh
# Mac integration, so nothing needs a terminal afterwards:
#   ./make-app.sh
# * ~/Applications/Remove Background.app: open it from Launchpad, Spotlight or
#   the Dock. Starts the server if it is not running (serve.sh) and opens
#   http://127.0.0.1:8777 in Chrome (default browser if Chrome is missing).
# * Finder Quick Action "Remove Background": right-click photos > Quick
#   Actions. Writes <name>.cutout.png next to each file (hr-matting,
#   transparent) through the same server, so the warm model is reused and the
#   results show up in the web UI's history as well.
# Both point at this folder: re-run after moving the repo. Safe to re-run.
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
APP="$HOME/Applications/Remove Background.app"
WF="$HOME/Library/Services/Remove Background.workflow"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -x .venv/bin/python ] || ./setup.sh
chmod +x serve.sh

# ----------------------------------------------------------------- the app
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
  <key>CFBundleVersion</key><string>2</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/launch" <<SH
#!/bin/sh
"$DIR/serve.sh" || exit 1
open -a "Google Chrome" "http://127.0.0.1:8777" 2>/dev/null || open "http://127.0.0.1:8777"
SH
chmod +x "$APP/Contents/MacOS/launch"

# icon: dark rounded square, checkerboard "removed" half, white subject
ICONSET="$TMP/AppIcon.iconset"
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
touch "$APP"
echo "built: $APP"

# -------------------------------------------------------- the Quick Action
# An Automator workflow of the "Run Shell Script" kind; Finder passes the
# selected files as arguments. Written with plistlib rather than by hand so
# the script needs no XML escaping.
cat > "$TMP/qa.sh" <<'SH'
DIR="__DIR__"
URL="http://127.0.0.1:8777"
"$DIR/serve.sh" || exit 1
ok=0; failed=0
for f in "$@"; do
  out="${f%.*}.cutout.png"
  if /usr/bin/curl -s -f -m 900 -F "image=@\"${f//\"/\\\"}\"" -F model=hr-matting \
       -o "$out" "$URL/api/cutout"; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1)); rm -f "$out"
  fi
done
/usr/bin/osascript -e "display notification \"$ok done, $failed failed\" with title \"Remove Background\""
SH
sed -i '' "s|__DIR__|$DIR|" "$TMP/qa.sh"

rm -rf "$WF"
mkdir -p "$WF/Contents"
.venv/bin/python - "$TMP/qa.sh" "$WF" <<'PY'
import plistlib, sys, uuid
script = open(sys.argv[1]).read()
wf = sys.argv[2]
u = lambda: str(uuid.uuid4()).upper()
params = [("inputMethod", 0), ("shell", "/bin/sh"), ("source", ""),
          ("COMMAND_STRING", ""), ("CheckedForUserDefaultShell", False)]
action = {
    "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.string"]},
    "AMActionVersion": "2.0.3",
    "AMApplication": ["Automator"],
    "AMParameterProperties": {n: {} for n, _ in params},
    "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
    "ActionBundlePath": "/System/Library/Automator/Run Shell Script.action",
    "ActionName": "Run Shell Script",
    "ActionParameters": {"COMMAND_STRING": script, "CheckedForUserDefaultShell": True,
                         "inputMethod": 1, "shell": "/bin/zsh", "source": ""},
    "BundleIdentifier": "com.apple.RunShellScript",
    "CFBundleVersion": "2.0.3",
    "CanShowSelectedItemsWhenRun": False,
    "CanShowWhenRun": True,
    "Category": ["AMCategoryUtilities"],
    "Class Name": "RunShellScriptAction",
    "InputUUID": u(), "OutputUUID": u(), "UUID": u(),
    "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
    "UnlocalizedApplications": ["Automator"],
    "arguments": {str(i): {"default value": dv, "name": n, "required": "0",
                           "type": "0", "uuid": str(i)}
                  for i, (n, dv) in enumerate(params)},
    "isViewVisible": 1,
    "location": "309.000000:253.000000",
    "nibPath": "/System/Library/Automator/Run Shell Script.action/Contents/Resources/Base.lproj/main.nib",
}
doc = {
    "AMApplicationBuild": "523", "AMApplicationVersion": "2.10", "AMDocumentVersion": "2",
    "actions": [{"action": action, "isViewVisible": 1}],
    "connectors": {},
    "workflowMetaData": {
        "applicationBundleIDsByPath": {}, "applicationPaths": [],
        "inputTypeIdentifier": "com.apple.Automator.fileSystemObject.image",
        "outputTypeIdentifier": "com.apple.Automator.nothing",
        "presentationMode": 15, "processesInput": 0,
        "serviceApplicationBundleID": "com.apple.finder",
        "serviceApplicationPath": "/System/Library/CoreServices/Finder.app",
        "serviceInputTypeIdentifier": "com.apple.Automator.fileSystemObject.image",
        "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
        "serviceProcessesInput": 0,
        "useAutomaticInputType": 0,
        "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
    },
}
info = {"NSServices": [{
    "NSBackgroundColorName": "background",
    "NSIconName": "NSActionTemplate",
    "NSMenuItem": {"default": "Remove Background"},
    "NSMessage": "runWorkflowAsService",
    "NSRequiredContext": {"NSApplicationIdentifier": "com.apple.finder"},
    "NSSendFileTypes": ["public.image"],
}]}
with open(f"{wf}/Contents/document.wflow", "wb") as f:
    plistlib.dump(doc, f)
with open(f"{wf}/Contents/Info.plist", "wb") as f:
    plistlib.dump(info, f)
PY
/System/Library/CoreServices/pbs -update 2>/dev/null || true
echo "built: $WF"
echo
echo "open Remove Background from Launchpad or Spotlight; drag it to the Dock to keep it there."
echo "in Finder, right-click a photo > Quick Actions > Remove Background."
