"""
Drag-and-drop web UI for rmbg.py. A local remove.bg, on localhost.

    ./run-ui.sh            # or: .venv/bin/python server.py

Then open http://127.0.0.1:8777 and drop photos on the page. The browser only
ever talks to 127.0.0.1; nothing leaves the machine.

Memory: a model loads on the first image that asks for it and is dropped
again after --idle seconds without work for that model (default 60). Several
models can sit in memory at once (weights are ~0.5 GB each; the 4-16 GB peak
is activations, released after every image), so switching models to compare
does not reload anything. A batch of photos in a row runs on the warm model.
Loading costs ~1-4 s, unloading ~0.3 s. The x on a model chip (POST
/api/unload with model=...) drops one model, "Free all" drops every one.

History: each result (original + cutout + settings) is kept under
~/Library/Application Support/remove-bg/history for --history-days days
(default 7, 0 keeps nothing on disk), so the page shows it again after a
reload or a server restart, and Redo can rerun an old photo with another
model. Delete on a card removes it from disk at once.

Sleep: after --sleep-after minutes without an image (default 10, 0 = never)
the process exec()s into sleeper.py on the same port: the idle torch runtime
alone is ~1 GB, the sleeper ~15 MB. Status polling from an open tab does not
count as work, so tabs can stay open forever. The page (or serve.sh, or a
fresh GET /) wakes it with POST /api/wake; the first image after that waits
~3 s for the import plus the usual model load. "Quit" (POST /api/quit)
stops the server for good, for the app-launched case with no terminal.
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

import rmbg
import sam

app = FastAPI(title="remove-bg (local)")

STATE = {"device": "auto", "fp32": False, "idle": 60, "last_used": {},
         "size": None, "history_days": 7, "sleep_after": 10,
         "last_work": time.time()}
GPU = threading.Lock()   # one inference at a time; also guards load/unload
CANCELLED = {}           # job id -> time the client gave up on it

SLEEPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleeper.py")
HISTORY_DIR = os.path.expanduser("~/Library/Application Support/remove-bg/history")
BROWSER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}   # Chrome shows these; heic it does not

# UI label, one-line hint, approximate download in MB. Order = order in the menus.
MODEL_INFO = {
    "hr-matting": ("HR matting (best)", "Best edges: hair, fur, glass, smoke. 2048px, the slowest.", 430),
    "hr":         ("HR (crisp edges)", "Hard, clean edges at 2048px: products, logos, packshots.", 430),
    "matting":    ("matting 1024 (fast)", "Soft edges at 1024px, about 4x faster. Fine for web-size photos.", 430),
    "general":    ("general 1024", "All-round at 1024px with hard edges. Balanced speed and quality.", 430),
    "portrait":   ("portrait", "Tuned for people, 1024px.", 430),
    "lite":       ("lite (fastest)", "Small backbone, the fastest. Quick previews or a machine without GPU.", 170),
}


# ------------------------------------------------------------------ models

def touch(model):
    STATE["last_used"][model] = STATE["last_work"] = time.time()


def loaded_names():
    names = {key[0] for key in list(rmbg._LOADED)}
    if sam.loaded():
        names.add(sam.NAME)
    return names


def idle_for(model):
    return time.time() - STATE["last_used"].get(model, 0)


def unload_in(model):
    if STATE["idle"] <= 0:
        return None
    return max(0, int(STATE["idle"] - idle_for(model)))


def unload_one(model):
    if model == sam.NAME:
        sam.unload()
    else:
        rmbg.unload_model(model)


def loaded_models():
    """What is in memory right now, for /api/status: one entry per model."""
    out = []
    for (model, device, half), (_, size, _, _) in list(rmbg._LOADED.items()):
        out.append({"model": model, "device": device,
                    "precision": "fp16" if half else "fp32",
                    "size": STATE["size"] or size, "unload_in": unload_in(model)})
    if sam.loaded():
        out.append({"model": sam.NAME, "device": "cpu", "precision": "fp32",
                    "size": 1024, "unload_in": unload_in(sam.NAME)})
    return out


def repo_downloaded(repo):
    """(weights on disk?, MB in the hub cache) for one hugging face repo."""
    from huggingface_hub.constants import HF_HUB_CACHE

    d = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    snaps = os.path.join(d, "snapshots")
    if not os.path.isdir(snaps):
        return False, 0
    weights = any(f.endswith(".safetensors")
                  for s in os.listdir(snaps)
                  for f in os.listdir(os.path.join(snaps, s)))
    blobs = os.path.join(d, "blobs")
    total = sum(os.path.getsize(os.path.join(blobs, f))
                for f in os.listdir(blobs)) if os.path.isdir(blobs) else 0
    return weights, total // 2**20


def idle_reaper():
    """Background thread: unload each model once it sat unused for --idle s."""
    while True:
        time.sleep(2)
        if STATE["idle"] <= 0:
            continue
        for model in loaded_names():
            if idle_for(model) < STATE["idle"]:
                continue
            with GPU:
                if model in loaded_names() and idle_for(model) >= STATE["idle"]:
                    unload_one(model)
                    print(f"{model} unloaded after idle", flush=True)


def sleep_reaper():
    """Background thread: after --sleep-after minutes without an image,
    become sleeper.py (same PID, same port, ~15 MB instead of ~1 GB)."""
    while True:
        time.sleep(15)
        mins = STATE["sleep_after"]
        if mins <= 0 or GPU.locked():
            continue
        if time.time() - STATE["last_work"] >= mins * 60:
            print(f"no work for {mins} min, going to sleep", flush=True)
            os.execv(sys.executable, [sys.executable, SLEEPER, *sys.argv[1:]])


# ----------------------------------------------------------------- history

def history_dir(hid):
    if not re.fullmatch(r"[0-9a-f]{32}", hid):
        raise HTTPException(404)
    d = os.path.join(HISTORY_DIR, hid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        raise HTTPException(404)
    return d


def history_meta(hid):
    with open(os.path.join(history_dir(hid), "meta.json")) as f:
        return json.load(f)


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def history_save(raw, name, png, meta, cut=None):
    """Write original + result + meta; meta.json goes last so a half-written
    entry is invisible to history_list(). `cut` is the transparent RGBA
    cutout when `png` was flattened onto a background: the brush needs it."""
    hid = uuid.uuid4().hex
    d = os.path.join(HISTORY_DIR, hid)
    os.makedirs(d)
    ext = os.path.splitext(name or "")[1].lower() or ".img"
    with open(os.path.join(d, "orig" + ext), "wb") as f:
        f.write(raw)
    with open(os.path.join(d, "out.png"), "wb") as f:
        f.write(png)
    if cut is not None:
        with open(os.path.join(d, "cut.png"), "wb") as f:
            f.write(cut)
    meta = dict(meta, id=hid, name=name, ext=ext, ts=time.time())
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return hid


def history_list():
    """Every kept result, oldest first."""
    items = []
    if not os.path.isdir(HISTORY_DIR):
        return items
    for hid in os.listdir(HISTORY_DIR):
        try:
            with open(os.path.join(HISTORY_DIR, hid, "meta.json")) as f:
                items.append(json.load(f))
        except (OSError, ValueError):
            continue
    items.sort(key=lambda m: m["ts"])
    return items


def parse_points(points):
    """Form field -> [[x, y, label], ...] or raise HTTPException(400)."""
    try:
        pts = json.loads(points or "[]")
        assert isinstance(pts, list)
        out = []
        for p in pts:
            x, y, l = p
            assert all(isinstance(v, (int, float)) and v == v for v in (x, y))
            assert l in (0, 1)
            out.append([float(x), float(y), int(l)])
        return out
    except (ValueError, TypeError, AssertionError):
        raise HTTPException(400, "points must be [[x, y, 0|1], ...]")


def history_purge():
    days = STATE["history_days"]
    cutoff = time.time() - days * 86400
    for m in history_list():
        if days <= 0 or m["ts"] < cutoff:
            shutil.rmtree(os.path.join(HISTORY_DIR, m["id"]), ignore_errors=True)


def purge_reaper():
    while True:
        time.sleep(3600)
        history_purge()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>remove background — local</title>
<style>
  :root {
    --bg: #f6f6f4; --panel: #fff; --ink: #1a1a19; --muted: #6f6f6a;
    --line: #e2e2dd; --accent: #1a1a19; --drop: #ececE6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17171a; --panel: #1f1f23; --ink: #f0f0ee; --muted: #9a9a95;
      --line: #33333a; --accent: #f0f0ee; --drop: #26262b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif;
  }
  header {
    padding: 28px 24px 8px; max-width: 1080px; margin: 0 auto;
  }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; }
  main { max-width: 1080px; margin: 0 auto; padding: 16px 24px 64px; }
  #drop {
    border: 1.5px dashed var(--line); border-radius: 14px; background: var(--drop);
    padding: 46px 24px; text-align: center; cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  #drop.hot { border-color: var(--accent); background: var(--panel); }
  #drop b { display: block; font-size: 16px; margin-bottom: 4px; }
  #drop span { color: var(--muted); font-size: 13px; }
  .bar {
    display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
    margin: 18px 0 6px; font-size: 13px; color: var(--muted);
  }
  .bar label { display: flex; gap: 7px; align-items: center; }
  select, input[type=color] {
    font: inherit; color: var(--ink); background: var(--panel);
    border: 1px solid var(--line); border-radius: 7px; padding: 4px 7px;
  }
  input[type=color] { padding: 2px; width: 34px; height: 28px; }
  .swatches { display: flex; gap: 6px; }
  .sw {
    width: 26px; height: 26px; border-radius: 6px; cursor: pointer; padding: 0;
    border: 1px solid color-mix(in srgb, var(--ink) 28%, transparent);
  }
  .sw[aria-pressed=true] { outline: 2px solid var(--accent); outline-offset: 1px; }
  .sw.alpha {
    background-image:
      linear-gradient(45deg, #bbb 25%, transparent 25%, transparent 75%, #bbb 75%),
      linear-gradient(45deg, #bbb 25%, #fff 25%, #fff 75%, #bbb 75%);
    background-size: 12px 12px; background-position: 0 0, 6px 6px;
  }
  .hint { font-size: 12px; color: var(--muted); margin: 0 0 8px; }
  .cards { display: grid; gap: 14px; margin-top: 18px; }
  .card {
    position: relative;
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px; display: grid; grid-template-columns: 1fr 1fr auto; gap: 14px;
    align-items: center;
  }
  .corner {
    position: absolute; top: 6px; right: 6px; width: 24px; height: 24px; padding: 0;
    border: 0; border-radius: 50%; background: transparent; color: var(--muted);
    font: 18px/24px inherit; cursor: pointer;
  }
  .corner:hover { background: var(--drop); color: #c0392b; }
  @media (max-width: 720px) { .card { grid-template-columns: 1fr 1fr; } }
  .shot {
    aspect-ratio: 4 / 3; border-radius: 10px; overflow: hidden;
    display: grid; place-items: center; background: #00000010;
  }
  .shot.checker {
    background-image:
      linear-gradient(45deg, #00000018 25%, transparent 25%, transparent 75%, #00000018 75%),
      linear-gradient(45deg, #00000018 25%, transparent 25%, transparent 75%, #00000018 75%);
    background-size: 18px 18px; background-position: 0 0, 9px 9px;
  }
  .shot img { max-width: 100%; max-height: 100%; display: block; }
  .meta { grid-column: 1 / -1; display: flex; justify-content: space-between;
          font-size: 12px; color: var(--muted); }
  .actions { display: flex; flex-direction: column; gap: 8px; width: 236px; }
  .actions .row { display: flex; gap: 8px; }
  .actions .row > * { flex: 1 1 0; min-width: 0; }
  .actions .row > select { flex: 1.6 1 0; }
  .actions button.ghost { padding: 7px 8px; font-size: 13px; }
  .actions select { font-size: 12px; padding: 4px 4px; }
  .actions button.cancel { width: 100%; }
  button.go {
    font: inherit; border: 0; border-radius: 8px; padding: 8px 14px; cursor: pointer;
    background: var(--accent); color: var(--panel);
  }
  button.go[disabled] { opacity: .45; cursor: default; }
  .spin {
    width: 22px; height: 22px; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--accent);
    animation: r .7s linear infinite;
  }
  @keyframes r { to { transform: rotate(360deg); } }
  .err { color: #c0392b; font-size: 13px; }
  .status {
    display: flex; gap: 12px; align-items: center; margin-top: 10px; flex-wrap: wrap;
    font-size: 12px; color: var(--muted);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--line); }
  .dot.on { background: #3cb371; }
  button.ghost {
    font: inherit; font-size: 12px; cursor: pointer; padding: 3px 9px;
    border: 1px solid var(--line); border-radius: 6px;
    background: transparent; color: var(--ink);
  }
  button.ghost[disabled] { opacity: .4; cursor: default; }
  button.del:hover, button.cancel:hover { color: #c0392b; border-color: #c0392b; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    display: inline-flex; gap: 6px; align-items: center; padding: 2px 3px 2px 9px;
    border: 1px solid var(--line); border-radius: 999px; background: var(--panel);
  }
  .chip b { color: var(--ink); font-weight: 600; }
  .chip .x {
    font: inherit; line-height: 1; border: 0; background: transparent;
    color: var(--muted); cursor: pointer; padding: 3px 6px; border-radius: 999px;
  }
  .chip .x:hover { background: var(--drop); color: var(--ink); }
  .chip .x[disabled] { opacity: .35; cursor: default; }
  .tag { color: var(--ink); font-weight: 600; }

  /* click-to-select overlay */
  #picker {
    position: fixed; inset: 0; z-index: 10; background: rgba(12, 12, 14, .96);
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    padding: 16px 24px; color: #eee;
  }
  #picker[hidden] { display: none; }
  .pick-top {
    width: 100%; max-width: 1200px; display: flex; gap: 18px; align-items: center;
    justify-content: space-between; flex-wrap: wrap; font-size: 13px;
  }
  .pick-top .ghost { color: #eee; border-color: #555; }
  .pick-top .ghost[aria-pressed=true] { background: #eee; color: #111; border-color: #eee; }
  .pick-top .go { background: #4a8dff; color: #fff; padding: 6px 16px; }
  .pick-tools { display: flex; gap: 8px; align-items: center; }
  #picker-msg { font-weight: 600; min-width: 220px; }
  .pick-stage { position: relative; line-height: 0; }
  #picker-img { max-width: 94vw; max-height: calc(100vh - 130px); display: block; user-select: none; }
  #picker-canvas { position: absolute; inset: 0; cursor: crosshair; }
  .pick-hint { font-size: 12px; color: #999; }
  .ovl-close {
    width: 34px; height: 34px; padding: 0; margin-left: 6px; border-radius: 50%;
    border: 1px solid #555; background: transparent; color: #eee;
    font: 22px/32px inherit; cursor: pointer;
  }
  .ovl-close:hover { background: #333; border-color: #888; }

  /* brush overlay: same frame as the picker, scrollable zoomable stage */
  #brush {
    position: fixed; inset: 0; z-index: 10; background: rgba(12, 12, 14, .96);
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    padding: 16px 24px; color: #eee;
  }
  #brush[hidden] { display: none; }
  #brush label { display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: #bbb; }
  #brush input[type=range] { width: 90px; }
  .br-wrap { position: relative; line-height: 0; }
  .br-stage { overflow: auto; max-width: 94vw; max-height: calc(100vh - 130px); }
  #br-work {
    display: block; cursor: none;
    background-image:
      linear-gradient(45deg, #444 25%, transparent 25%, transparent 75%, #444 75%),
      linear-gradient(45deg, #444 25%, #666 25%, #666 75%, #444 75%);
    background-size: 20px 20px; background-position: 0 0, 10px 10px;
  }
  #br-work.white { background: #fff; }
  #br-work.black { background: #000; }
  #br-ring { position: absolute; inset: 0; pointer-events: none; }
</style>
</head>
<body>
<header>
  <h1>remove background</h1>
  <div class="sub">BiRefNet, running on this machine. Nothing is uploaded anywhere.</div>
</header>
<main>
  <div id="drop">
    <b>Drop images here</b>
    <span>or click to pick · jpg, png, webp, heic · paste with ⌘V</span>
    <input id="file" type="file" accept="image/*" multiple hidden>
  </div>

  <div class="bar">
    <label>Background
      <div class="swatches" id="sw">
        <button class="sw alpha" data-bg="" title="transparent" aria-pressed="true"></button>
        <button class="sw" data-bg="#ffffff" style="background:#fff" title="white"></button>
        <button class="sw" data-bg="#000000" style="background:#000" title="black"></button>
        <button class="sw" data-bg="#e9e4dc" style="background:#e9e4dc" title="warm grey"></button>
      </div>
    </label>
    <label>Custom <input type="color" id="pick" value="#3f8cff"></label>
    <label>Model <select id="model"></select></label>
    <label><input type="checkbox" id="tta"> extra pass (slower, cleaner)</label>
  </div>
  <div class="hint" id="hint"></div>

  <div class="status">
    <span id="stat">checking…</span>
    <div class="chips" id="chips"></div>
    <button class="ghost" id="free" disabled>Free all</button>
    <button class="ghost" id="wake" hidden>Wake up</button>
    <button class="ghost" id="quit">Quit</button>
  </div>

  <div class="cards" id="cards"></div>
</main>

<div id="picker" hidden>
  <div class="pick-top">
    <span id="picker-msg">Click the object you want to keep</span>
    <span class="pick-tools">
      <button class="ghost tool" data-label="1" aria-pressed="true">+ keep</button>
      <button class="ghost tool" data-label="0" aria-pressed="false">− exclude</button>
      <button class="ghost" id="picker-undo" disabled>Undo</button>
      <button class="ghost" id="picker-reset" disabled>Reset</button>
    </span>
    <span class="pick-tools">
      <button class="go" id="picker-go" disabled>Redo with selection</button>
      <button class="ovl-close" id="picker-close" title="close without applying (Esc)">×</button>
    </span>
  </div>
  <div class="pick-stage"><img id="picker-img" draggable="false"><canvas id="picker-canvas"></canvas></div>
  <div class="pick-hint">click = keep · ⌥-click or right-click = exclude · ⌘Z undo · Enter applies · Esc closes</div>
</div>

<div id="brush" hidden>
  <div class="pick-top">
    <span id="br-msg">Paint over what to bring back or erase</span>
    <span class="pick-tools">
      <button class="ghost br-tool" data-mode="restore" aria-pressed="true" title="paint the original back (R)">Restore</button>
      <button class="ghost br-tool" data-mode="erase" aria-pressed="false" title="make transparent (E)">Erase</button>
      <label>size <input type="range" id="br-size" min="2" max="400" value="40"></label>
      <label>soft <input type="range" id="br-soft" min="0" max="100" value="50"></label>
      <button class="ghost" id="br-undo" disabled>Undo</button>
      <button class="ghost" id="br-redo" disabled>Redo</button>
      <button class="ghost" id="br-reset" disabled>Reset</button>
    </span>
    <span class="pick-tools">
      <button class="ghost br-view" data-view="" aria-pressed="true" title="view on checkerboard">▦</button>
      <button class="ghost br-view" data-view="white" aria-pressed="false" title="view on white">white</button>
      <button class="ghost br-view" data-view="black" aria-pressed="false" title="view on black">black</button>
      <button class="ghost" id="br-zoomout" title="zoom out">−</button>
      <button class="ghost" id="br-zoomfit" title="fit to screen">fit</button>
      <button class="ghost" id="br-zoomin" title="zoom in">+</button>
      <button class="go" id="br-go" disabled>Apply changes</button>
      <button class="ovl-close" id="br-close" title="close without applying (Esc)">×</button>
    </span>
  </div>
  <div class="br-wrap">
    <div class="br-stage" id="br-stage"><canvas id="br-work"></canvas></div>
    <canvas id="br-ring"></canvas>
  </div>
  <div class="pick-hint">drag to paint · [ ] brush size · X swaps Restore/Erase · ⌘Z undo, ⌘⇧Z redo · Enter applies · Esc closes</div>
</div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const cards = document.getElementById('cards');
const stat = document.getElementById('stat');
const chips = document.getElementById('chips');
const free = document.getElementById('free');
const modelSel = document.getElementById('model');
const ttaBox = document.getElementById('tta');
const hint = document.getElementById('hint');
const pick = document.getElementById('pick');
let bg = '';
let MODELS = [];
const LABEL = {};
let busy = 0;
const working = {};   // model -> images in flight

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---- settings live in localStorage: model, background, extra pass
const SKEY = 'rmbg.settings';
function saveSettings() {
  try {
    localStorage.setItem(SKEY, JSON.stringify(
      { model: modelSel.value, bg, tta: ttaBox.checked }));
  } catch (e) {}
}
function setBg(v) {
  bg = v;
  let hit = false;
  document.querySelectorAll('.sw').forEach(s => {
    const on = s.dataset.bg === v;
    s.setAttribute('aria-pressed', on); hit = hit || on;
  });
  if (!hit && v) pick.value = v;
}
function restoreSettings() {
  let s = {};
  try { s = JSON.parse(localStorage.getItem(SKEY)) || {}; } catch (e) {}
  if (s.model in LABEL) modelSel.value = s.model;
  ttaBox.checked = !!s.tta;
  setBg(s.bg || '');
  showHint();
}

document.getElementById('sw').addEventListener('click', e => {
  const b = e.target.closest('.sw'); if (!b) return;
  setBg(b.dataset.bg); saveSettings();
});
pick.addEventListener('input', e => { setBg(e.target.value); saveSettings(); });
modelSel.addEventListener('change', () => { showHint(); saveSettings(); });
ttaBox.addEventListener('change', saveSettings);

// ---- models: labels, hints, what is already on disk
function optionsHtml(selected) {
  return MODELS.map(m =>
    `<option value="${m.name}" title="${esc(m.hint)}"` +
    `${m.name === selected ? ' selected' : ''}>${esc(m.label)}</option>`).join('');
}
async function loadModels() {
  const cur = modelSel.value;
  const info = await (await fetch('/api/models')).json();
  MODELS = info.models; MODELS.sam = info.sam;
  MODELS.forEach(m => { LABEL[m.name] = m.label; });
  modelSel.innerHTML = optionsHtml(cur in LABEL ? cur : MODELS[0].name);
  showHint();
}
function showHint() {
  const m = MODELS.find(x => x.name === modelSel.value); if (!m) return;
  hint.textContent = m.hint + ' ' + (m.downloaded
    ? `Downloaded (${m.size_mb} MB).`
    : `Downloads ~${m.size_mb} MB on first use.`);
}

// ---- drop / pick / paste
drop.addEventListener('click', () => file.click());
file.addEventListener('change', () => { handle(file.files); file.value = ''; });
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => {
  e.preventDefault(); drop.classList.add('hot');
}));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => {
  e.preventDefault(); drop.classList.remove('hot');
}));
drop.addEventListener('drop', e => handle(e.dataTransfer.files));
window.addEventListener('paste', e => {
  const f = [...e.clipboardData.files]; if (f.length) handle(f);
});

function handle(files) {
  const opts = { model: modelSel.value, bg, tta: ttaBox.checked };
  [...files].filter(f => f.type.startsWith('image/'))
    .forEach(f => run({ file: f, name: f.name }, opts));
}

// ---- memory status. After 10 min without work the server swaps itself for
// a tiny sleeper on the same port; dropping a photo (or Wake up) brings it back.
const wake = document.getElementById('wake');
async function refreshStatus() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (e) { stat.textContent = 'server not reachable'; return; }
  wake.hidden = !s.sleeping;
  if (s.sleeping) {
    chips.innerHTML = '';
    stat.textContent = 'server sleeping · memory freed, wakes up on the next image';
    free.disabled = true;
    return;
  }
  const loaded = new Set(s.loaded.map(m => m.model));
  const rows = s.loaded.map(m => {
    const w = working[m.model] > 0;
    const when = w ? 'working' : m.unload_in !== null ? `frees in ${m.unload_in}s` : 'kept';
    return `<span class="chip"><span class="dot on"></span><b>${esc(m.model)}</b>` +
      ` ${m.precision} · ${m.size}px · ${when}` +
      `<button class="x" data-model="${esc(m.model)}" title="unload ${esc(m.model)}"` +
      `${w ? ' disabled' : ''}>×</button></span>`;
  });
  for (const [m, n] of Object.entries(working))
    if (n > 0 && !loaded.has(m))
      rows.push(`<span class="chip"><span class="dot"></span><b>${esc(m)}</b> loading…&nbsp;</span>`);
  chips.innerHTML = rows.join('');
  stat.textContent = s.loaded.length ? 'in memory:'
    : busy ? '' : `memory free · a model loads on the next image (${s.ram_gb} GB RAM)`;
  free.disabled = busy > 0 || s.loaded.length === 0;
}
chips.addEventListener('click', async e => {
  const b = e.target.closest('.x'); if (!b) return;
  b.disabled = true;
  const body = new FormData();
  body.append('model', b.dataset.model);
  await fetch('/api/unload', { method: 'POST', body });
  refreshStatus();
});
free.addEventListener('click', async () => {
  free.disabled = true;
  await fetch('/api/unload', { method: 'POST' });
  refreshStatus();
});
async function ensureAwake() {
  let s = null;
  try { s = await (await fetch('/api/status')).json(); } catch (e) {}
  if (s && !s.sleeping) return;
  if (!s) throw new Error('server not running: open Remove Background again');
  fetch('/api/wake', { method: 'POST' }).catch(() => {});
  for (let i = 0; i < 90; i++) {          // torch import takes a few seconds
    await new Promise(r => setTimeout(r, 700));
    try {
      s = await (await fetch('/api/status')).json();
      if (!s.sleeping) return;
    } catch (e) {}
  }
  throw new Error('server did not wake up');
}
wake.addEventListener('click', async () => {
  wake.disabled = true; stat.textContent = 'waking up…';
  try { await ensureAwake(); } catch (e) { stat.textContent = e.message; }
  wake.disabled = false; refreshStatus();
});
let statusTimer = null;
document.getElementById('quit').addEventListener('click', async () => {
  if (busy && !confirm('Images are still processing. Quit anyway?')) return;
  clearInterval(statusTimer);
  try { await fetch('/api/quit', { method: 'POST' }); } catch (e) {}
  document.querySelector('main').innerHTML =
    '<p class="sub">Server stopped. You can close this tab; open Remove Background again to start it.</p>';
});

// ---- cards. One per (image, model) run; results also live in the server's
// history, so a reload or a restart shows them again.
function cardEl(name, opts) {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="shot"><img></div>
    <div class="shot checker"><div class="spin"></div></div>
    <div class="actions">
      <div class="row">
        <button class="go save" disabled title="download the PNG">Save</button>
        <button class="ghost copy" disabled title="copy the PNG to the clipboard">Copy</button>
      </div>
      <div class="row">
        <button class="ghost pickobj" disabled title="click on the object to keep, for photos with several things in them">Pick object</button>
        <button class="ghost touchup" disabled title="brush: bring parts back or erase them by hand">Touch up</button>
      </div>
      <div class="row">
        <select class="redo-model" title="model for Redo">${optionsHtml(opts.model)}</select>
        <button class="ghost redo" disabled title="run this photo again with the model on the left">Redo</button>
      </div>
      <button class="ghost cancel">Cancel</button>
    </div>
    <button class="corner del" hidden title="delete this result">×</button>
    <div class="meta">
      <span>${esc(name)} · <span class="tag">${esc(LABEL[opts.model] || opts.model)}</span>` +
      `${opts.tta ? ' · extra pass' : ''}` +
      `${opts.points && opts.points.length ? ` · picked object (${opts.points.length} click${opts.points.length > 1 ? 's' : ''})` : ''}` +
      `${opts.edited ? ' · brush' : ''}</span>
      <span class="t">working…</span>
    </div>`;
  return card;
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = old; }, 1500);
}

// src: {file, name} for a photo from this session, {id, name} for one from
// history. out: {id, url, blob?, seconds}. Redo of a stored result reads the
// original back from the server, so it works after a reload too.
function finish(card, src, opts, out) {
  const [before, after] = card.querySelectorAll('.shot');
  card.dataset.id = out.id || '';
  after.innerHTML = `<img src="${out.url}" loading="lazy">`;
  if (opts.bg) after.classList.remove('checker');
  if (out.id && src.file) {
    // heic: Chrome cannot show the original, the server sends a jpeg preview
    const b = before.querySelector('img');
    const orig = `/api/history/${out.id}/orig`;
    b.onerror = () => { b.onerror = null; b.src = orig; };
    if (b.complete && !b.naturalWidth) b.src = orig;
  }
  card.querySelector('.t').textContent = out.seconds.toFixed(1) + ' s';
  card.querySelector('.cancel').hidden = true;
  card.querySelector('.del').hidden = false;

  const stem = src.name.replace(/\.[^.]+$/, '');
  const save = card.querySelector('.save');
  save.disabled = false;
  save.onclick = () => {
    const a = document.createElement('a');
    a.href = out.url; a.download = `${stem}.${opts.model}.cutout.png`; a.click();
  };
  const copy = card.querySelector('.copy');
  copy.disabled = false;
  copy.onclick = async () => {
    try {
      const blob = out.blob || (out.blob = await (await fetch(out.url)).blob());
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      flash(copy, 'Copied');
    } catch (e) { flash(copy, 'Copy failed'); }
  };
  const redo = card.querySelector('.redo');
  if (src.file || out.id) {
    redo.disabled = false;
    redo.onclick = () => run(src.file ? src : { id: out.id, name: src.name },
      { ...opts, model: card.querySelector('.redo-model').value }, card);
  }
  const pickBtn = card.querySelector('.pickobj');
  if (out.id) {   // needs the stored original: the picker clicks on it
    pickBtn.disabled = false;
    pickBtn.onclick = () => openPicker({ id: out.id, name: src.name },
      { ...opts, model: card.querySelector('.redo-model').value }, card);
    const tu = card.querySelector('.touchup');
    tu.disabled = false;
    tu.onclick = () => openBrush({ id: out.id, name: src.name }, opts, card);
  }
  card.querySelector('.del').onclick = async () => {
    if (out.id) await fetch(`/api/history/${out.id}`, { method: 'DELETE' });
    card.remove();
  };
}

// extra = {url, blob, strokes}: post a brush-edited PNG to /api/edit instead
// of running the model; the card flow is the same.
async function run(src, opts, anchor, extra) {
  const card = cardEl(src.name, opts);
  if (anchor) anchor.before(card); else cards.prepend(card);
  const [before, after] = card.querySelectorAll('.shot');
  before.querySelector('img').src =
    src.file ? URL.createObjectURL(src.file) : `/api/history/${src.id}/orig`;

  // cancel: drop the request client-side and tell the server to skip the
  // job if it is still waiting for the GPU
  const job = crypto.randomUUID();
  const ctrl = new AbortController();
  card.querySelector('.cancel').onclick = () => {
    ctrl.abort();
    const b = new FormData(); b.append('job', job);
    fetch('/api/cancel', { method: 'POST', body: b });
  };

  const body = new FormData();
  if (src.file) body.append('image', src.file); else body.append('source', src.id);
  body.append('bg', opts.bg);
  body.append('model', opts.model);
  body.append('tta', opts.tta ? '1' : '');
  body.append('job', job);
  if (opts.points && opts.points.length) body.append('points', JSON.stringify(opts.points));
  if (extra && extra.blob) { body.append('image', extra.blob, 'edit.png'); body.append('strokes', extra.strokes); }

  const t0 = performance.now();
  busy++; working[opts.model] = (working[opts.model] || 0) + 1; refreshStatus();
  try {
    await ensureAwake();
    if (ctrl.signal.aborted) throw new DOMException('cancelled', 'AbortError');
    const res = await fetch(extra && extra.url || '/api/cutout', { method: 'POST', body, signal: ctrl.signal });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    finish(card, src, opts, {
      id: res.headers.get('X-Id') || '', url: URL.createObjectURL(blob), blob,
      seconds: (performance.now() - t0) / 1000,
    });
    const m = MODELS.find(x => x.name === opts.model);
    if (m && !m.downloaded) loadModels();   // first use just fetched the weights
  } catch (err) {
    if (err.name === 'AbortError') { card.remove(); return; }
    after.innerHTML = '<div class="err"></div>';
    after.querySelector('.err').textContent = err.message;
    card.querySelector('.t').textContent = 'failed';
    card.querySelector('.cancel').hidden = true;
    const del = card.querySelector('.del');
    del.hidden = false; del.onclick = () => card.remove();
  } finally {
    busy--; working[opts.model]--; refreshStatus();
  }
}

async function loadHistory() {
  let h;
  try { h = await (await fetch('/api/history')).json(); } catch (e) { return; }
  for (const it of h.items) {           // oldest first; prepend puts newest on top
    const opts = { model: it.model, bg: it.bg, tta: it.tta, points: it.points || [], edited: !!it.edited };
    const card = cardEl(it.name, opts);
    cards.prepend(card);
    card.querySelector('.shot img').src = `/api/history/${it.id}/orig`;
    finish(card, { id: it.id, name: it.name }, opts,
      { id: it.id, url: `/api/history/${it.id}/out`, seconds: it.seconds });
  }
}

// ---- click-to-select overlay (SAM 2). Clicks are in original image pixels;
// the server returns a mask preview, Redo with selection runs BiRefNet on the picked
// object as a new card above the one it came from.
const pk = {
  el: document.getElementById('picker'), img: document.getElementById('picker-img'),
  canvas: document.getElementById('picker-canvas'), msg: document.getElementById('picker-msg'),
  go: document.getElementById('picker-go'), undo: document.getElementById('picker-undo'),
  reset: document.getElementById('picker-reset'),
  src: null, opts: null, anchor: null, points: [], mask: null, label: 1, seq: 0,
};

function openPicker(src, opts, anchor) {
  Object.assign(pk, { src, opts, anchor, points: [], mask: null, seq: pk.seq + 1 });
  pk.img.src = `/api/history/${src.id}/orig`;
  pk.el.hidden = false;
  const sam = MODELS.sam || {};
  pk.msg.textContent = 'Click the object you want to keep';
  document.querySelector('.pick-hint').textContent = (sam.downloaded ? '' :
    `first use downloads SAM 2 (~${sam.size_mb} MB) · `) +
    'click = keep · ⌥-click or right-click = exclude · ⌘Z undo · Enter applies · Esc closes';
  pickButtons();
  if (opts.points && opts.points.length) {      // re-open a picked card: start from its clicks
    pk.points = opts.points.map(p => [...p]);
    pk.img.decode().then(() => { sizeCanvas(); refreshMask(); }).catch(() => {});
  }
}
function closePicker() { pk.el.hidden = true; pk.img.src = ''; pk.mask = null; }
function pickButtons() {
  const n = pk.points.length;
  pk.undo.disabled = pk.reset.disabled = n === 0;
  pk.go.disabled = !(n && pk.mask);
}
function sizeCanvas() {
  const r = pk.img.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  pk.canvas.width = Math.round(r.width * dpr); pk.canvas.height = Math.round(r.height * dpr);
  pk.canvas.style.width = r.width + 'px'; pk.canvas.style.height = r.height + 'px';
  drawPick();
}
function drawPick() {
  const c = pk.canvas, ctx = c.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  if (pk.mask) {
    // kept = full brightness with a green outline, removed = dimmed hard
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(0,0,0,.72)'; ctx.fillRect(0, 0, c.width, c.height);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.drawImage(pk.mask, 0, 0, c.width, c.height);
    // outline: the mask in green, grown by d px, minus the mask itself
    const d = Math.max(2, Math.round(c.width / 500));
    const shape = document.createElement('canvas'); shape.width = c.width; shape.height = c.height;
    const sc = shape.getContext('2d');
    sc.drawImage(pk.mask, 0, 0, c.width, c.height);
    sc.globalCompositeOperation = 'source-in'; sc.fillStyle = '#3ddc84'; sc.fillRect(0, 0, c.width, c.height);
    const ring = document.createElement('canvas'); ring.width = c.width; ring.height = c.height;
    const rc = ring.getContext('2d');
    for (const [dx, dy] of [[d, 0], [-d, 0], [0, d], [0, -d], [d, d], [-d, -d], [d, -d], [-d, d]])
      rc.drawImage(shape, dx, dy);
    rc.globalCompositeOperation = 'destination-out'; rc.drawImage(shape, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(ring, 0, 0);
  }
  const sx = c.width / pk.img.naturalWidth, sy = c.height / pk.img.naturalHeight;
  const rad = 6 * (window.devicePixelRatio || 1);
  for (const [x, y, l] of pk.points) {
    ctx.beginPath(); ctx.arc(x * sx, y * sy, rad, 0, 7);
    ctx.fillStyle = l ? '#3ddc84' : '#ff5252'; ctx.fill();
    ctx.lineWidth = rad / 3; ctx.strokeStyle = '#fff'; ctx.stroke();
  }
}
async function refreshMask() {
  const seq = ++pk.seq;
  if (!pk.points.length) { pk.mask = null; drawPick(); pickButtons(); pk.msg.textContent = 'Click the object you want to keep'; return; }
  pk.msg.textContent = pk.mask ? 'Selecting…' : 'Reading the image…';
  const body = new FormData();
  body.append('source', pk.src.id); body.append('points', JSON.stringify(pk.points));
  try {
    await ensureAwake();
    const res = await fetch('/api/sam', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const img = new Image();
    img.src = URL.createObjectURL(await res.blob());
    await img.decode();
    if (seq !== pk.seq) return;                 // a newer click already answered
    pk.mask = img;
    pk.msg.textContent = 'Bright with green outline = kept, dark = removed · click to add, ⌥-click to exclude';
  } catch (e) {
    if (seq !== pk.seq) return;
    pk.msg.textContent = 'Failed: ' + e.message;
  }
  drawPick(); pickButtons();
}
function pickAt(e, label) {
  const r = pk.canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) / r.width * pk.img.naturalWidth);
  const y = Math.round((e.clientY - r.top) / r.height * pk.img.naturalHeight);
  pk.points.push([x, y, label]);
  drawPick(); pickButtons(); refreshMask();
}
pk.canvas.addEventListener('click', e => pickAt(e, e.altKey ? 0 : pk.label));
pk.canvas.addEventListener('contextmenu', e => { e.preventDefault(); pickAt(e, 0); });
document.querySelectorAll('#picker .tool').forEach(b => b.addEventListener('click', () => {
  pk.label = +b.dataset.label;
  document.querySelectorAll('#picker .tool').forEach(t => t.setAttribute('aria-pressed', t === b));
}));
pk.undo.addEventListener('click', () => { pk.points.pop(); refreshMask(); });
pk.reset.addEventListener('click', () => { pk.points = []; refreshMask(); });
document.getElementById('picker-close').addEventListener('click', closePicker);
pk.go.addEventListener('click', () => {
  const { src, opts, anchor, points } = pk;
  closePicker();
  run(src, { ...opts, points: points.map(p => [...p]) }, anchor);
});
pk.img.addEventListener('load', sizeCanvas);
window.addEventListener('resize', () => { if (!pk.el.hidden) sizeCanvas(); });
window.addEventListener('keydown', e => {
  if (pk.el.hidden) return;
  if (e.key === 'Escape') closePicker();
  else if (e.key === 'Enter' && !pk.go.disabled) pk.go.click();
  else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); pk.undo.click(); }
});

// ---- brush overlay. Works on the full-resolution cutout in a canvas:
// Restore stamps the original's pixels back through a soft circle, Erase
// cuts alpha with destination-out. Strokes are kept as data and replayed
// from the untouched cutout for Undo, so no pixel snapshots pile up.
const br = {
  el: document.getElementById('brush'), stage: document.getElementById('br-stage'),
  work: document.getElementById('br-work'), ring: document.getElementById('br-ring'),
  msg: document.getElementById('br-msg'), go: document.getElementById('br-go'),
  undo: document.getElementById('br-undo'), redo: document.getElementById('br-redo'),
  reset: document.getElementById('br-reset'), sizeEl: document.getElementById('br-size'),
  softEl: document.getElementById('br-soft'),
  src: null, opts: null, anchor: null, base: null, orig: null,
  strokes: [], undone: [], cur: null, mode: 'restore', zoom: 1, fit: 1,
  tmp: document.createElement('canvas'), mouse: null,
};

async function loadImage(url) {
  const img = new Image(); img.src = url; await img.decode(); return img;
}
async function openBrush(src, opts, anchor) {
  Object.assign(br, { src, opts, anchor, strokes: [], undone: [], cur: null, mouse: null });
  br.el.hidden = false;
  br.msg.textContent = 'Loading…';
  try {
    const [cut, orig] = await Promise.all([
      loadImage(`/api/history/${src.id}/cut`), loadImage(`/api/history/${src.id}/orig`)]);
    br.base = cut;
    br.orig = document.createElement('canvas');
    br.orig.width = cut.naturalWidth; br.orig.height = cut.naturalHeight;
    br.orig.getContext('2d').drawImage(orig, 0, 0, cut.naturalWidth, cut.naturalHeight);
    br.work.width = cut.naturalWidth; br.work.height = cut.naturalHeight;
    br.sizeEl.max = Math.max(50, Math.round(cut.naturalWidth / 4));
    br.sizeEl.value = Math.max(8, Math.round(cut.naturalWidth / 40));
    setZoom('fit');
    replay();
    br.msg.textContent = 'Paint over what to bring back or erase';
  } catch (e) { br.msg.textContent = 'Failed: ' + e.message; }
}
function closeBrush() {
  if (br.strokes.length && !confirm('Discard the brush changes?')) return;
  br.el.hidden = true; br.base = br.orig = null; br.strokes = []; br.work.width = 1;
}
function brushButtons() {
  br.undo.disabled = !br.strokes.length; br.redo.disabled = !br.undone.length;
  br.reset.disabled = !br.strokes.length; br.go.disabled = !br.strokes.length;
}
function setZoom(z) {
  const maxW = window.innerWidth * .94, maxH = window.innerHeight - 130;
  br.fit = Math.min(1, maxW / br.work.width, maxH / br.work.height);
  const steps = [br.fit, 1, 2, 4].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
  if (z === 'fit') br.zoom = br.fit;
  else if (z === 'in') br.zoom = steps.find(s => s > br.zoom + 1e-6) || br.zoom;
  else if (z === 'out') br.zoom = [...steps].reverse().find(s => s < br.zoom - 1e-6) || br.zoom;
  br.work.style.width = (br.work.width * br.zoom) + 'px';
  br.work.style.height = (br.work.height * br.zoom) + 'px';
  requestAnimationFrame(() => {
    const r = br.stage.getBoundingClientRect();
    br.ring.width = r.width; br.ring.height = r.height;
    br.ring.style.width = r.width + 'px'; br.ring.style.height = r.height + 'px';
    drawRing();
  });
}
function brushGrad(ctx, x, y, r, soft) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, 'rgba(0,0,0,1)');
  g.addColorStop(Math.max(0, 1 - soft), 'rgba(0,0,0,1)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  return g;
}
function stamp(ctx, x, y, s) {
  const r = s.size / 2;
  if (s.mode === 'erase') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = brushGrad(ctx, x, y, r, s.soft);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    return;
  }
  const x0 = Math.floor(x - r) - 1, y0 = Math.floor(y - r) - 1, n = Math.ceil(r * 2) + 3;
  const t = br.tmp; t.width = n; t.height = n;                  // resizing clears it
  const tc = t.getContext('2d');
  tc.fillStyle = brushGrad(tc, x - x0, y - y0, r, s.soft);
  tc.beginPath(); tc.arc(x - x0, y - y0, r, 0, 7); tc.fill();
  tc.globalCompositeOperation = 'source-in';
  tc.drawImage(br.orig, x0, y0, n, n, 0, 0, n, n);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(t, x0, y0);
}
function segment(ctx, a, b, s) {
  const step = Math.max(1, s.size / 6);
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]), k = Math.max(1, Math.ceil(d / step));
  for (let i = 1; i <= k; i++)
    stamp(ctx, a[0] + (b[0] - a[0]) * i / k, a[1] + (b[1] - a[1]) * i / k, s);
}
function replay() {
  const ctx = br.work.getContext('2d');
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, br.work.width, br.work.height);
  ctx.drawImage(br.base, 0, 0);
  for (const s of br.strokes) {
    stamp(ctx, s.pts[0][0], s.pts[0][1], s);
    for (let i = 1; i < s.pts.length; i++) segment(ctx, s.pts[i - 1], s.pts[i], s);
  }
  brushButtons();
}
function drawRing() {
  const c = br.ring, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!br.mouse) return;
  const r = br.sizeEl.value * br.zoom / 2;
  ctx.beginPath(); ctx.arc(br.mouse[0], br.mouse[1], r, 0, 7);
  ctx.lineWidth = 1.5; ctx.strokeStyle = br.mode === 'erase' ? '#ff5252' : '#3ddc84'; ctx.stroke();
  ctx.beginPath(); ctx.arc(br.mouse[0], br.mouse[1], r, 0, 7);
  ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = .5; ctx.stroke();
}
function brushPos(e) {
  const r = br.work.getBoundingClientRect();
  return [(e.clientX - r.left) / br.zoom, (e.clientY - r.top) / br.zoom];
}
br.work.addEventListener('pointerdown', e => {
  if (e.button !== 0) return;
  e.preventDefault();
  br.work.setPointerCapture(e.pointerId);
  const p = brushPos(e);
  br.cur = { mode: br.mode, size: +br.sizeEl.value, soft: br.softEl.value / 100, pts: [p] };
  stamp(br.work.getContext('2d'), p[0], p[1], br.cur);
});
br.work.addEventListener('pointermove', e => {
  if (!br.cur) return;
  const p = brushPos(e), last = br.cur.pts[br.cur.pts.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 1) return;
  segment(br.work.getContext('2d'), last, p, br.cur);
  br.cur.pts.push(p);
});
function endStroke() {
  if (!br.cur) return;
  br.strokes.push(br.cur); br.undone = []; br.cur = null;
  brushButtons();
}
br.work.addEventListener('pointerup', endStroke);
br.work.addEventListener('pointercancel', endStroke);
br.stage.addEventListener('pointermove', e => {
  const r = br.stage.getBoundingClientRect();
  br.mouse = [e.clientX - r.left, e.clientY - r.top]; drawRing();
});
br.stage.addEventListener('pointerleave', () => { br.mouse = null; drawRing(); });
function setBrushMode(m) {
  br.mode = m;
  document.querySelectorAll('.br-tool').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === m));
  drawRing();
}
document.querySelectorAll('.br-tool').forEach(b => b.addEventListener('click', () => setBrushMode(b.dataset.mode)));
document.querySelectorAll('.br-view').forEach(b => b.addEventListener('click', () => {
  br.work.className = b.dataset.view;
  document.querySelectorAll('.br-view').forEach(v => v.setAttribute('aria-pressed', v === b));
}));
br.sizeEl.addEventListener('input', drawRing);
br.undo.addEventListener('click', () => { if (br.strokes.length) { br.undone.push(br.strokes.pop()); replay(); } });
br.redo.addEventListener('click', () => { if (br.undone.length) { br.strokes.push(br.undone.pop()); replay(); } });
br.reset.addEventListener('click', () => { br.strokes = []; br.undone = []; replay(); });
document.getElementById('br-zoomin').addEventListener('click', () => setZoom('in'));
document.getElementById('br-zoomout').addEventListener('click', () => setZoom('out'));
document.getElementById('br-zoomfit').addEventListener('click', () => setZoom('fit'));
document.getElementById('br-close').addEventListener('click', closeBrush);
br.go.addEventListener('click', async () => {
  br.go.disabled = true; br.msg.textContent = 'Saving…';
  const blob = await new Promise(r => br.work.toBlob(r, 'image/png'));
  const { src, opts, anchor, strokes } = br;
  br.strokes = [];            // applied, nothing to discard
  closeBrush();
  run(src, { ...opts, edited: true }, anchor, { url: '/api/edit', blob, strokes: strokes.length });
});
window.addEventListener('resize', () => { if (!br.el.hidden) setZoom(br.zoom === br.fit ? 'fit' : 'same'); });
window.addEventListener('keydown', e => {
  if (br.el.hidden) return;
  if (e.key === 'Escape') closeBrush();
  else if (e.key === 'Enter' && !br.go.disabled) br.go.click();
  else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); (e.shiftKey ? br.redo : br.undo).click(); }
  else if (e.key === '[') { br.sizeEl.value = Math.max(2, br.sizeEl.value / 1.25); drawRing(); }
  else if (e.key === ']') { br.sizeEl.value = Math.min(br.sizeEl.max, br.sizeEl.value * 1.25); drawRing(); }
  else if (e.key === 'x' || e.key === 'X') setBrushMode(br.mode === 'erase' ? 'restore' : 'erase');
  else if (e.key === 'e' || e.key === 'E') setBrushMode('erase');
  else if (e.key === 'r' || e.key === 'R') setBrushMode('restore');
});

(async () => {
  await loadModels();
  restoreSettings();
  refreshStatus();
  statusTimer = setInterval(refreshStatus, 3000);
  loadHistory();
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/status")
def api_status():
    return {"loaded": loaded_models(), "idle": STATE["idle"],
            "ram_gb": round(rmbg.total_ram_gb())}


@app.get("/api/models")
def api_models():
    out = []
    for name, (label, hint, mb) in MODEL_INFO.items():
        have, size = repo_downloaded(rmbg.MODELS[name][0])
        out.append({"name": name, "label": label, "hint": hint,
                    "native": rmbg.MODELS[name][1],
                    "downloaded": have, "size_mb": size if have else mb})
    have, size = repo_downloaded(sam.MODEL_ID)
    return {"models": out,
            "sam": {"downloaded": have, "size_mb": size if have else 180}}


@app.post("/api/unload")
def api_unload(model: str = Form("")):
    """Drop one model (form field `model`) or, without it, every model."""
    with GPU:
        if model:
            unload_one(model)
        else:
            rmbg.unload_models()
            sam.unload()
    return {"loaded": loaded_models()}


@app.post("/api/sam")
def api_sam(source: str = Form(""), points: str = Form("")):
    """Mask preview for the click-to-select overlay: the stored original
    under history id `source`, clicks as JSON [[x, y, label], ...] in image
    pixels. Returns a downscaled RGBA PNG, blue with the mask as alpha.
    CPU only, so it does not queue behind the GPU."""
    pts = parse_points(points)
    if not pts:
        return Response("no points", status_code=400)
    meta = history_meta(source)
    image = Image.open(os.path.join(history_dir(source), "orig" + meta["ext"])).convert("RGB")
    touch(sam.NAME)
    logits = sam.segment(image, pts, key=source)
    touch(sam.NAME)
    buf = io.BytesIO()
    sam.preview_png(logits).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/api/wake")
def api_wake():
    return {"waking": False}   # already awake; the sleeper answers this for real


@app.post("/api/quit")
def api_quit():
    # exit a moment later so this response still reaches the browser
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"quit": True}


@app.post("/api/cancel")
def api_cancel(job: str = Form("")):
    now = time.time()
    if job:
        CANCELLED[job] = now
    for k in [k for k, t in CANCELLED.items() if now - t > 3600]:
        del CANCELLED[k]
    return {"cancelled": job}


@app.get("/api/history")
def api_history():
    return {"items": history_list(), "days": STATE["history_days"]}


@app.get("/api/history/{hid}/orig")
def api_history_orig(hid: str):
    d = history_dir(hid)
    meta = history_meta(hid)
    path = os.path.join(d, "orig" + meta["ext"])
    if meta["ext"] in BROWSER_EXT:
        return FileResponse(path)
    preview = os.path.join(d, "preview.jpg")   # heic & co: browsers cannot show them
    if not os.path.isfile(preview):
        Image.open(path).convert("RGB").save(preview, quality=88)
    return FileResponse(preview)


@app.get("/api/history/{hid}/out")
def api_history_out(hid: str):
    return FileResponse(os.path.join(history_dir(hid), "out.png"))


@app.get("/api/history/{hid}/cut")
def api_history_cut(hid: str):
    """The transparent cutout, even when the result was flattened onto a colour."""
    d = history_dir(hid)
    cut = os.path.join(d, "cut.png")
    return FileResponse(cut if os.path.isfile(cut) else os.path.join(d, "out.png"))


@app.post("/api/edit")
def api_edit(source: str = Form(""), image: UploadFile = File(...), strokes: int = Form(0)):
    """A cutout retouched with the brush in the browser: store it as a new
    history entry next to its source (same original, background, model)."""
    meta = history_meta(source)
    rgba = Image.open(io.BytesIO(image.file.read())).convert("RGBA")
    with open(os.path.join(history_dir(source), "orig" + meta["ext"]), "rb") as f:
        raw = f.read()
    bg = meta.get("bg", "")
    out = rmbg.flatten(rgba, rmbg.parse_background(bg)) if bg else rgba
    png = png_bytes(out)
    headers = {}
    if STATE["history_days"] > 0:
        headers["X-Id"] = history_save(raw, meta["name"], png, {
            "model": meta["model"], "bg": bg, "tta": meta.get("tta", False),
            "points": meta.get("points", []), "edited": True, "strokes": strokes,
            "seconds": 0, "width": rgba.width, "height": rgba.height},
            cut=png_bytes(rgba) if bg else None)
    return Response(png, media_type="image/png", headers=headers)


@app.delete("/api/history/{hid}")
def api_history_delete(hid: str):
    shutil.rmtree(history_dir(hid), ignore_errors=True)
    return {"deleted": hid}


@app.post("/api/cutout")
def api_cutout(
    image: UploadFile | None = File(None),
    source: str = Form(""),        # history id, instead of an upload (Redo)
    bg: str = Form(""),
    model: str = Form("hr-matting"),
    tta: str = Form(""),
    job: str = Form(""),           # client id, so /api/cancel can skip it
    points: str = Form(""),        # JSON [[x, y, label], ...]: keep only the clicked object
):
    # sync endpoint on purpose: FastAPI runs it in a worker thread, so the
    # event loop keeps answering /api/status while the GPU is busy.
    if model not in rmbg.MODELS:
        return Response(f"unknown model {model!r}", status_code=400)
    if image is not None:
        raw, name = image.file.read(), image.filename or "image"
    elif source:
        meta = history_meta(source)
        name = meta["name"]
        with open(os.path.join(history_dir(source), "orig" + meta["ext"]), "rb") as f:
            raw = f.read()
    else:
        return Response("no image", status_code=400)
    src = Image.open(io.BytesIO(raw)).convert("RGB")
    pts = parse_points(points)

    t0 = time.time()
    with GPU:
        if job in CANCELLED:
            return Response("cancelled", status_code=499)
        touch(model)
        net, size, device, half = rmbg.load_model(
            model, rmbg.pick_device(STATE["device"]),
            half=False if STATE["fp32"] else None)
        size = STATE["size"] or size
        if pts:
            touch(sam.NAME)
            try:
                rgba, _ = sam.cutout_picked(src, pts, net, size, device, half=half,
                                            tta=bool(tta), key=source or None)
            except ValueError as e:
                return Response(str(e), status_code=422)
            touch(sam.NAME)
        else:
            rgba, _ = rmbg.cutout(src, net, size, device, half=half, tta=bool(tta))
        touch(model)

    if bg:
        out = rmbg.flatten(rgba, rmbg.parse_background(bg))
    else:
        out = rgba

    png = png_bytes(out)
    headers = {"X-Seconds": f"{time.time() - t0:.2f}"}
    if STATE["history_days"] > 0 and job not in CANCELLED:
        headers["X-Id"] = history_save(raw, name, png, {
            "model": model, "bg": bg, "tta": bool(tta), "points": pts,
            "seconds": round(time.time() - t0, 2),
            "width": src.width, "height": src.height},
            cut=png_bytes(rgba) if bg else None)
    return Response(png, media_type="image/png", headers=headers)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("-m", "--model", default="hr-matting", choices=list(rmbg.MODELS),
                    help="checkpoint to warm up at startup")
    ap.add_argument("-s", "--size", type=int,
                    help="inference resolution (default follows RAM)")
    ap.add_argument("--idle", type=int, default=60,
                    help="unload a model after this many idle seconds, 0 = never")
    ap.add_argument("--sleep-after", type=int, default=10,
                    help="swap to the ~15 MB sleeper.py after this many minutes "
                         "without an image, 0 = never")
    ap.add_argument("--history-days", type=int, default=7,
                    help="keep results on disk this long, 0 = keep nothing")
    ap.add_argument("--no-warmup", action="store_true",
                    help="do not load the model at startup, wait for the first image")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    STATE.update(device=args.device, fp32=args.fp32, idle=args.idle, size=args.size,
                 sleep_after=args.sleep_after, history_days=args.history_days,
                 last_work=time.time())
    history_purge()
    if not args.no_warmup:
        print(f"warming up {rmbg.MODELS[args.model][0]} ...", flush=True)
        rmbg.load_model(args.model, rmbg.pick_device(args.device),
                        half=False if args.fp32 else None)
        touch(args.model)
    for m in loaded_models():
        print(f"{m['model']}: {m['precision']} at {m['size']}px on {m['device']}, "
              f"{rmbg.total_ram_gb():.0f} GB RAM, idle unload after {args.idle}s")
    for fn in (idle_reaper, sleep_reaper, purge_reaper):
        threading.Thread(target=fn, daemon=True).start()
    print(f"\n  open  http://{args.host}:{args.port}\n", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
