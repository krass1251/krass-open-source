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
(default 7, 0 writes nothing new), so the page shows it again after a
reload or a server restart, and Redo can rerun an old photo with another
model. Delete on a card removes it from disk at once.

Edit on a card opens a full-screen editor over the stored original and
cutout: SAM 2 clicks to select an object, a brush to restore or erase by
hand, and Generate to rerun the model with the clicks (a draft, so tries
never become cards). Done posts the composed pixels to /api/edit, which
stores them as one new card next to the one they came from.

Sleep: after --sleep-after minutes without an image (default 10, 0 = never)
the process exec()s into sleeper.py on the same port: the idle torch runtime
alone is ~1 GB, the sleeper ~15 MB. Any request except status polling counts
as work (a reload, the editor), and the swap waits for requests in
flight, so an open tab can stay open forever and never loses a request.

Only the page on this machine may talk to the API: Host must be loopback and
Origin, when a browser sends one, must match it. Without that any web page
in the browser could POST /api/quit or queue GPU work (simple form posts
skip the CORS preflight), and a DNS-rebound name could read the history. The page (or serve.sh, or a
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
import traceback
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageColor

import rmbg
import sam

app = FastAPI(title="remove-bg (local)")

STATE = {"device": "auto", "fp32": False, "idle": 60, "last_used": {},
         "size": None, "history_days": 7, "sleep_after": 10,
         "last_work": time.time(), "host": "127.0.0.1"}
GPU = threading.Lock()   # one inference at a time; also guards load/unload
INFLIGHT = [0]           # requests being served right now; sleep waits for zero
INFLIGHT_LOCK = threading.Lock()
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
CANCELLED = {}           # job id -> time the client gave up on it
JOBS = {}                # job id -> stage, for the card's status line

SETTINGS_PATH = os.path.expanduser("~/Library/Application Support/remove-bg/settings.json")


def load_settings():
    """Model, background and extra pass as last set on the page. The page
    and the Finder Quick Action both follow them."""
    try:
        with open(SETTINGS_PATH) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def parse_bg(value):
    """Background from the page or settings.json -> RGB tuple, None for
    transparent. Colours only (the CLI's parse_background also takes file
    paths, which a web request must not), and a bad value is a 400 before
    any model has run."""
    if not value:
        return None
    try:
        return ImageColor.getrgb(str(value))[:3]
    except ValueError:
        raise HTTPException(400, f"unknown background colour {value!r}")


def local_request(request):
    """Host must be loopback (unless --host opened the server up on purpose)
    and Origin, when the browser sends one, must be the same host."""
    try:
        host = urlsplit("//" + request.headers.get("host", "")).hostname or ""
        origin = request.headers.get("origin")
        origin_host = urlsplit(origin).hostname or "" if origin else host
    except ValueError:
        return False
    if STATE["host"] not in ("0.0.0.0", "::") and host not in LOCAL_HOSTS:
        return False
    return origin_host == host


@app.middleware("http")
async def guard(request: Request, call_next):
    if not local_request(request):
        return Response("forbidden: not a local request", status_code=403)
    with INFLIGHT_LOCK:
        INFLIGHT[0] += 1
    try:
        if request.url.path != "/api/status":
            STATE["last_work"] = time.time()
        return await call_next(request)
    finally:
        with INFLIGHT_LOCK:
            INFLIGHT[0] -= 1


def save_settings(d):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(d, f)

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
    try:   # a .DS_Store in snapshots/ or a vanished blob must not take the UI down
        weights = any(f.endswith(".safetensors")
                      for s in os.listdir(snaps)
                      if os.path.isdir(os.path.join(snaps, s))
                      for f in os.listdir(os.path.join(snaps, s)))
        blobs = os.path.join(d, "blobs")
        total = sum(os.path.getsize(os.path.join(blobs, f))
                    for f in os.listdir(blobs)
                    if os.path.isfile(os.path.join(blobs, f))) if os.path.isdir(blobs) else 0
    except OSError:
        return False, 0
    return weights, total // 2**20


def forever(step, every):
    """Background thread: step() every `every` seconds. An exception is
    printed and the loop goes on; a dead reaper would silently leave models
    in memory, the server awake or the history unpurged."""
    while True:
        time.sleep(every)
        try:
            step()
        except Exception:
            traceback.print_exc()


def idle_step():
    """Unload each model once it sat unused for --idle s."""
    if STATE["idle"] <= 0:
        return
    for model in loaded_names():
        if idle_for(model) < STATE["idle"]:
            continue
        with GPU:
            if model in loaded_names() and idle_for(model) >= STATE["idle"]:
                unload_one(model)
                print(f"{model} unloaded after idle", flush=True)


def sleep_step():
    """After --sleep-after minutes without work, become sleeper.py (same
    PID, same port, ~15 MB instead of ~1 GB). Never while a request is
    being served: exec() would cut it off mid-flight."""
    mins = STATE["sleep_after"]
    if mins <= 0 or GPU.locked() or INFLIGHT[0]:
        return
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
    cutout when `png` was flattened onto a background: the editor needs it."""
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
                m = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(m, dict) and m.get("id") == hid and "ts" in m:
            items.append(m)
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
    """Drop entries older than --history-days. 0 means nothing new is
    written; what earlier runs left on disk stays until deleted on the page,
    a flag flip must not wipe a week of results."""
    days = STATE["history_days"]
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    for m in history_list():
        if m["ts"] < cutoff:
            shutil.rmtree(os.path.join(HISTORY_DIR, m["id"]), ignore_errors=True)


def purge_step():
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
  .cards-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 22px; font-size: 12px; color: var(--muted);
  }
  .cards-head[hidden] { display: none; }
  .cards { display: grid; gap: 14px; margin-top: 10px; }
  .card[hidden] { display: none; }
  #toast {
    position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%); z-index: 5;
    display: flex; gap: 12px; align-items: center; padding: 8px 10px 8px 16px;
    background: var(--ink); color: var(--panel); border-radius: 10px; font-size: 13px;
    box-shadow: 0 6px 24px rgba(0,0,0,.25);
  }
  #toast[hidden] { display: none; }
  #toast .ghost { color: var(--panel); border-color: color-mix(in srgb, var(--panel) 40%, transparent); }
  .card {
    position: relative;
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px 14px;
    align-items: center;
  }
  .corner {
    position: absolute; top: 6px; right: 6px; width: 24px; height: 24px; padding: 0;
    border: 0; border-radius: 50%; background: transparent; color: var(--muted);
    font: 18px/24px inherit; cursor: pointer;
  }
  .corner:hover { background: var(--drop); color: #c0392b; }
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
  /* card: meta line on top, before/after, then the actions in working order */
  .meta { grid-column: 1 / -1; display: flex; justify-content: space-between; gap: 12px;
          padding-right: 26px; font-size: 12px; color: var(--muted); }
  .actions { grid-column: 1 / -1; display: flex; gap: 8px 18px; flex-wrap: wrap; align-items: center; }
  .actions .grp { display: flex; gap: 8px; align-items: center; }
  .actions .grp.end { margin-left: auto; }
  .actions button.ghost { padding: 7px 12px; font-size: 13px; }
  .actions button.go { padding: 7px 18px; font-size: 13px; }
  .actions select { font-size: 12px; padding: 5px 6px; }
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

  .ovl-close {
    width: 34px; height: 34px; padding: 0; margin-left: 6px; border-radius: 50%;
    border: 1px solid #555; background: transparent; color: #eee;
    font: 22px/32px inherit; cursor: pointer;
  }
  .ovl-close:hover { background: #333; border-color: #888; }

  /* editor overlay: tools in a left panel, one big stage for the image */
  #editor {
    position: fixed; inset: 0; z-index: 10; background: rgba(12, 12, 14, .97);
    color: #eee; padding: 14px 18px; gap: 12px 16px;
    display: grid; grid-template-columns: 236px 1fr; grid-template-rows: auto 1fr auto;
  }
  #editor[hidden] { display: none; }
  #editor > * { min-width: 0; min-height: 0; }   /* or the stage grows instead of scrolling */
  #editor .ghost { color: #eee; border-color: #555; }
  #editor .ghost[aria-pressed=true] { background: #eee; color: #111; border-color: #eee; }
  #editor .go { background: #4a8dff; color: #fff; padding: 6px 16px; }
  #editor select {
    width: 100%; font-size: 12px; color: #eee; background: #2a2a30; border-color: #555;
  }
  #editor label { display: flex; gap: 6px; align-items: center; font-size: 12px; color: #bbb; }
  #editor input[type=range] { width: 100px; }
  .ed-top {
    grid-column: 1 / -1; display: flex; gap: 18px; align-items: center;
    justify-content: space-between; font-size: 13px;
  }
  #ed-msg { font-weight: 600; }
  .ed-panel { overflow: auto; display: flex; flex-direction: column; gap: 14px; }
  .ed-sec { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .ed-sec[hidden] { display: none; }
  .ed-sec h3 {
    width: 100%; margin: 0; font: 600 10px/1.4 inherit; letter-spacing: .09em;
    text-transform: uppercase; color: #888;
  }
  .ed-stage-box { position: relative; }
  .ed-stage { position: absolute; inset: 0; overflow: auto; background: #0d0d0f; border-radius: 10px; }
  .ed-wrap { position: relative; margin: auto; cursor: crosshair; }
  #editor.brush .ed-wrap { cursor: none; }
  .ed-wrap canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .ed-wrap canvas[hidden] { display: none; }
  .checkers {
    background-image:
      linear-gradient(45deg, #444 25%, transparent 25%, transparent 75%, #444 75%),
      linear-gradient(45deg, #444 25%, #666 25%, #666 75%, #444 75%);
    background-size: 20px 20px; background-position: 0 0, 10px 10px;
  }
  #ed-work.white { background: #fff; }
  #ed-work.black { background: #000; }
  #ed-handle {
    position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px;
    background: #fff; cursor: ew-resize; touch-action: none;
  }
  #ed-handle[hidden] { display: none; }
  #ed-handle::after {
    content: '↔'; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 30px; height: 30px; border-radius: 50%; background: #fff; color: #111;
    font-size: 16px; line-height: 30px; text-align: center;
  }
  #ed-ring { position: absolute; inset: 0; pointer-events: none; }
  #ed-versions { display: flex; flex-direction: column; gap: 6px; width: 100%; }
  .ed-ver { display: flex; gap: 8px; align-items: center; width: 100%; text-align: left; }
  .ed-ver canvas { width: 34px; height: 34px; border-radius: 4px; flex: none; background-size: 10px 10px; }
  .ed-hint { grid-column: 1 / -1; font-size: 12px; color: #999; }
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

  <div class="cards-head" id="cards-head" hidden>
    <span id="count"></span>
    <button class="ghost" id="clear" title="delete every finished result (with Undo)">Clear all</button>
  </div>
  <div class="cards" id="cards"></div>
</main>

<div id="toast" hidden><span>Deleted</span><button class="ghost" id="undo-del">Undo</button></div>

<div id="editor" hidden>
  <div class="ed-top">
    <span id="ed-msg" class="t">Loading…</span>
    <span class="ed-sec">
      <button class="go" id="ed-done" disabled title="save the result as a new card (Enter)">Done</button>
      <button class="ovl-close" id="ed-close" title="close without saving (Esc)">×</button>
    </span>
  </div>
  <div class="ed-panel">
    <div class="ed-sec">
      <h3>Tool</h3>
      <button class="ghost ed-tool" data-tool="pick" aria-pressed="true" title="click the object to keep (P)">Pick</button>
      <button class="ghost ed-tool" data-tool="brush" aria-pressed="false" title="paint by hand (B)">Brush</button>
    </div>
    <div class="ed-sec" id="ed-pick">
      <h3>Selection</h3>
      <button class="ghost ed-label" data-label="1" aria-pressed="true">+ keep</button>
      <button class="ghost ed-label" data-label="0" aria-pressed="false">− exclude</button>
      <button class="ghost" id="ed-sel-erase" disabled title="make the selected region transparent, no model run">Erase selection</button>
      <button class="ghost" id="ed-sel-restore" disabled title="paint the original back inside the selection">Restore selection</button>
      <button class="ghost" id="ed-clear" disabled>Clear clicks</button>
    </div>
    <div class="ed-sec" id="ed-brush" hidden>
      <h3>Brush</h3>
      <button class="ghost ed-mode" data-mode="restore" aria-pressed="true" title="paint the original back (R)">Restore</button>
      <button class="ghost ed-mode" data-mode="erase" aria-pressed="false" title="make transparent (E)">Erase</button>
      <label>size <input type="range" id="ed-size" min="2" max="400" value="40"></label>
      <label>soft <input type="range" id="ed-soft" min="0" max="100" value="50"></label>
    </div>
    <div class="ed-sec">
      <h3>View</h3>
      <button class="ghost ed-view" data-view="orig" aria-pressed="false" title="the original (1)">Original</button>
      <button class="ghost ed-view" data-view="result" aria-pressed="true" title="the cutout (2)">Result</button>
      <button class="ghost ed-view" data-view="compare" aria-pressed="false" title="drag the slider (3)">Compare</button>
      <button class="ghost ed-bd" data-bd="" aria-pressed="true" title="on a checkerboard">▦</button>
      <button class="ghost ed-bd" data-bd="white" aria-pressed="false" title="on white">white</button>
      <button class="ghost ed-bd" data-bd="black" aria-pressed="false" title="on black">black</button>
      <button class="ghost" id="ed-zoomout" title="zoom out">−</button>
      <button class="ghost" id="ed-zoomfit" title="fit to the stage">fit</button>
      <button class="ghost" id="ed-zoomin" title="zoom in">+</button>
    </div>
    <div class="ed-sec">
      <h3>Generate</h3>
      <select id="ed-model" title="model for Generate"></select>
      <label><input type="checkbox" id="ed-tta"> extra pass</label>
      <button class="go" id="ed-gen" title="rerun the model with these clicks">Generate</button>
    </div>
    <div class="ed-sec">
      <h3>History</h3>
      <button class="ghost" id="ed-undo" disabled>Undo</button>
      <button class="ghost" id="ed-redo" disabled>Redo</button>
      <button class="ghost" id="ed-reset" disabled>Reset</button>
    </div>
    <div class="ed-sec">
      <h3>Versions</h3>
      <div id="ed-versions"></div>
    </div>
  </div>
  <div class="ed-stage-box">
    <div class="ed-stage" id="ed-stage">
      <div class="ed-wrap" id="ed-wrap">
        <canvas id="ed-orig" hidden></canvas>
        <canvas id="ed-work" class="checkers"></canvas>
        <canvas id="ed-mask"></canvas>
        <div id="ed-handle" hidden></div>
      </div>
    </div>
    <canvas id="ed-ring"></canvas>
  </div>
  <div class="ed-hint" id="ed-hint"></div>
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

// ---- settings (model, background, extra pass) live on the server, so the
// Finder Quick Action follows them too
function saveSettings() {
  const body = JSON.stringify({ model: modelSel.value, bg, tta: ttaBox.checked });
  // the sleeper answers 503 and forgets, so wake the server first
  ensureAwake()
    .then(() => fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }))
    .then(async r => { if (!r.ok) throw new Error(await r.text()); })
    .catch(e => { stat.textContent = 'settings not saved: ' + e.message; });
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
async function restoreSettings() {
  let s = {};
  try { s = await (await fetch('/api/settings')).json(); } catch (e) {}
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
const JOBS = {};   // job id -> status element, so the server's stage lands on the right line
const STAGE = {
  queued: 'waiting for the GPU…', download: 'downloading the model, first use (a few minutes)…',
  load: 'loading the model…', sam: 'loading SAM 2…', run: 'processing…',
};
async function refreshStatus() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (e) { stat.textContent = 'server not reachable'; return; }
  for (const [job, stage] of Object.entries(s.jobs || {})) {
    const t = JOBS[job];
    if (t) t.textContent = STAGE[stage] || stage;
  }
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
    <div class="meta">
      <span>${esc(name)} · <span class="tag">${esc(LABEL[opts.model] || opts.model)}</span>` +
      `${opts.tta ? ' · extra pass' : ''}` +
      `${opts.points && opts.points.length ? ` · picked object (${opts.points.length} click${opts.points.length > 1 ? 's' : ''})` : ''}` +
      `${opts.edited ? ' · edited' : ''}</span>
      <span class="t">working…</span>
    </div>
    <button class="corner del" hidden title="delete this result">×</button>
    <div class="shot"><img></div>
    <div class="shot checker"><div class="spin"></div></div>
    <div class="actions">
      <div class="grp">
        <select class="redo-model" title="model for Redo">${optionsHtml(opts.model)}</select>
        <button class="ghost redo" disabled title="run this photo again with the model on the left">Redo</button>
      </div>
      <div class="grp">
        <button class="ghost edit" disabled title="pick the object, brush it by hand, rerun the model">Edit</button>
      </div>
      <div class="grp end">
        <button class="ghost cancel">Cancel</button>
        <button class="ghost copy" disabled title="copy the PNG to the clipboard">Copy</button>
        <button class="go save" disabled title="download the PNG">Save</button>
      </div>
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
      { ...opts, model: card.querySelector('.redo-model').value, edited: false }, card);
  }
  const editBtn = card.querySelector('.edit');
  if (out.id) {   // needs the stored original and cutout: the editor works on both
    editBtn.disabled = false;
    editBtn.onclick = () => openEditor({ id: out.id, name: src.name }, opts, card,
      card.querySelector('.redo-model').value);
  }
  card.querySelector('.del').onclick = () => softDelete([card]);
}

// ---- delete with undo: cards disappear at once, the server is told 5 s
// later unless Undo is hit. Clear all goes through the same path.
const trash = { cards: [], timer: null, el: document.getElementById('toast') };
function softDelete(list) {
  list.forEach(c => { c.hidden = true; trash.cards.push(c); });
  clearTimeout(trash.timer);
  trash.timer = setTimeout(commitDeletes, 5000);
  const n = trash.cards.length;
  trash.el.querySelector('span').textContent = n === 1 ? 'Deleted' : `${n} deleted`;
  trash.el.hidden = false;
  updateHead();
}
async function commitDeletes(leaving) {
  const list = trash.cards;
  trash.cards = []; trash.el.hidden = true;
  // the sleeper answers 503 and forgets; wake the server unless the tab is
  // closing, where there is no time to wait for it
  if (!leaving) { try { await ensureAwake(); } catch (e) {} }
  for (const c of list) {
    if (!c.dataset.id) { c.remove(); continue; }
    fetch(`/api/history/${c.dataset.id}`, { method: 'DELETE', keepalive: true })
      .then(r => { if (r.ok || r.status === 404) c.remove(); else throw new Error(r.status); })
      .catch(() => { c.hidden = false; updateHead(); });   // not deleted: show it again
  }
}
document.getElementById('undo-del').addEventListener('click', () => {
  clearTimeout(trash.timer);
  trash.cards.forEach(c => { c.hidden = false; });
  trash.cards = []; trash.el.hidden = true;
  updateHead();
});
window.addEventListener('pagehide', () => { if (trash.cards.length) commitDeletes(true); });
document.getElementById('clear').addEventListener('click', () => {
  const done = [...cards.querySelectorAll('.card:not([hidden])')].filter(c => c.querySelector('.cancel').hidden);
  if (done.length) softDelete(done);
});
function updateHead() {
  const n = cards.querySelectorAll('.card:not([hidden])').length;
  document.getElementById('cards-head').hidden = n === 0;
  document.getElementById('count').textContent = n === 1 ? '1 result' : `${n} results`;
}
new MutationObserver(updateHead).observe(cards, { childList: true });

function cancelJob(job) {   // the server skips it if it still waits for the GPU
  const b = new FormData(); b.append('job', job);
  fetch('/api/cancel', { method: 'POST', body: b });
}
// one request to the model server with the bookkeeping around it: busy and
// working counts for the chips, the job's stage line, a wake-up first.
// Resolves to the Response; a non-2xx throws with the server's text, a
// cancel with an AbortError.
async function submit(url, body, job, ctrl, tEl, model) {
  busy++; working[model] = (working[model] || 0) + 1; JOBS[job] = tEl; refreshStatus();
  try {
    tEl.textContent = 'starting…';
    await ensureAwake();
    tEl.textContent = url === '/api/cutout' ? 'sending…' : 'saving…';
    const res = await fetch(url, { method: 'POST', body, signal: ctrl.signal });
    if (!res.ok) throw new Error(await res.text());
    const m = MODELS.find(x => x.name === model);
    if (m && !m.downloaded) loadModels().catch(() => {});   // first use just fetched the weights
    return res;
  } finally {
    busy--; working[model]--; delete JOBS[job]; refreshStatus();
  }
}

// extra = {url, blob, strokes}: post an edited PNG to /api/edit instead of
// running the model; the card flow is the same. Resolves true on success.
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
  card.querySelector('.cancel').onclick = () => { ctrl.abort(); cancelJob(job); };

  const body = new FormData();
  if (src.file) body.append('image', src.file); else body.append('source', src.id);
  body.append('bg', opts.bg);
  body.append('model', opts.model);
  body.append('tta', opts.tta ? '1' : '');
  body.append('job', job);
  body.append('points', JSON.stringify(opts.points || []));   // "[]" = whole subject
  if (extra && extra.blob) { body.append('image', extra.blob, 'edit.png'); body.append('strokes', extra.strokes); }

  const t0 = performance.now();
  try {
    const res = await submit(extra && extra.url || '/api/cutout', body, job, ctrl,
                             card.querySelector('.t'), opts.model);
    const blob = await res.blob();
    finish(card, src, opts, {
      id: res.headers.get('X-Id') || '', url: URL.createObjectURL(blob), blob,
      seconds: (performance.now() - t0) / 1000,
    });
    return true;
  } catch (err) {
    if (err.name === 'AbortError') { card.remove(); return false; }
    after.innerHTML = '<div class="err"></div>';
    after.querySelector('.err').textContent = err.message;
    card.querySelector('.t').textContent = 'failed';
    card.querySelector('.cancel').hidden = true;
    const del = card.querySelector('.del');
    del.hidden = false; del.onclick = () => card.remove();
    return false;
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

// ---- editor overlay. One workspace per card: Pick (SAM 2 clicks, in
// original image pixels) and Brush share one undo stack, Generate reruns the
// model as a draft version without touching the card list, Done posts the
// composed pixels to /api/edit as a new card above the one it came from.
const ed = {
  el: document.getElementById('editor'), stage: document.getElementById('ed-stage'),
  wrap: document.getElementById('ed-wrap'), msg: document.getElementById('ed-msg'),
  orig: document.getElementById('ed-orig'), work: document.getElementById('ed-work'),
  maskC: document.getElementById('ed-mask'), ring: document.getElementById('ed-ring'),
  handle: document.getElementById('ed-handle'), hint: document.getElementById('ed-hint'),
  done: document.getElementById('ed-done'), gen: document.getElementById('ed-gen'),
  undoB: document.getElementById('ed-undo'), redoB: document.getElementById('ed-redo'),
  resetB: document.getElementById('ed-reset'), clearB: document.getElementById('ed-clear'),
  selErase: document.getElementById('ed-sel-erase'),
  selRestore: document.getElementById('ed-sel-restore'),
  modelEl: document.getElementById('ed-model'), ttaEl: document.getElementById('ed-tta'),
  sizeEl: document.getElementById('ed-size'), softEl: document.getElementById('ed-soft'),
  versEl: document.getElementById('ed-versions'),
  src: null, opts: null, anchor: null, W: 0, H: 0,
  versions: [],          // {label, img: ImageBitmap, blob, points, model, tta}
  basePoints: [],        // the card's own clicks: the starting point, not undoable
  acts: [], undone: [],  // one stack for clicks, strokes, selection ops, version switches
  open: 0,               // bumped on every open and close: async answers check it
  tool: 'pick', label: 1, mode: 'restore', view: 'result', bd: '',
  zoom: 1, fit: 1, compare: 50, cur: null, mouse: null,
  mask: null, seq: 0, pending: false,   // SAM mask for the current clicks, a fetch under way
  job: null, ctrl: null, saving: false,
  tmp: document.createElement('canvas'), tmp2: document.createElement('canvas'),
  layer: document.createElement('canvas'), layerFor: null,   // dim + outline, built once per mask
};

function edReset() {
  Object.assign(ed, { versions: [], acts: [], undone: [], basePoints: [], W: 0, H: 0,
                      compare: 50, cur: null, mouse: null, mask: null, pending: false,
                      job: null, ctrl: null, saving: false, layerFor: null });
}
async function fetchBitmap(url) {   // decoded once, straight from the response
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  return [await createImageBitmap(blob), blob];
}

// opts are the card's real options (what produced it); model only seeds the
// Generate select, so a card can be re-run with another model from here
async function openEditor(src, opts, anchor, model) {
  if (!ed.el.hidden) { closeEditor(); if (!ed.el.hidden) return; }   // one session at a time
  if (document.activeElement) document.activeElement.blur();   // Space must not re-click Edit
  const open = ++ed.open;
  edReset();
  Object.assign(ed, { src, opts, anchor, basePoints: (opts.points || []).map(p => [...p]) });
  ed.el.hidden = false;
  ed.msg.textContent = 'Loading…';
  ed.modelEl.innerHTML = optionsHtml(model || opts.model);
  ed.ttaEl.checked = !!opts.tta;
  setTool('pick'); setLabel(1); setBrushMode(ed.mode); setBd(ed.bd); setView('result');
  edVersions(); edButtons();
  try {
    await ensureAwake();         // asleep, the sleeper would serve a 503 instead of the images
    if (open !== ed.open) return;                  // closed while waking
    const [[cut, cutBlob], [orig]] = await Promise.all([
      fetchBitmap(`/api/history/${src.id}/cut`), fetchBitmap(`/api/history/${src.id}/orig`)]);
    if (open !== ed.open) { cut.close(); orig.close(); return; }
    const W = ed.W = cut.width, H = ed.H = cut.height;
    ed.orig.width = W; ed.orig.height = H;
    ed.orig.getContext('2d').drawImage(orig, 0, 0, W, H);
    orig.close();
    ed.work.width = W; ed.work.height = H;
    const s = Math.min(1, 2000 / Math.max(W, H));  // the overlay never needs full resolution
    ed.maskC.width = Math.round(W * s); ed.maskC.height = Math.round(H * s);
    ed.sizeEl.max = Math.max(50, Math.round(W / 4));
    ed.sizeEl.value = Math.max(8, Math.round(W / 40));
    ed.versions = [{ label: 'from card', img: cut, blob: cutBlob, points: ed.basePoints.map(p => [...p]),
                     model: opts.model, tta: !!opts.tta }];
    setZoom('fit');
    edReplay();
    ed.msg.textContent = edIdle();
    refreshMask();               // a picked card opens with its selection already showing
  } catch (e) {
    if (open === ed.open) ed.msg.textContent = 'Failed: ' + e.message;
  }
}
function closeEditor(force) {
  if (!force && ed.acts.length && !confirm('Discard the changes in the editor?')) return;
  if (ed.ctrl) { ed.ctrl.abort(); cancelJob(ed.job); }   // a Generate in flight
  ed.open++;                     // whatever is still in flight is stale now
  ed.el.hidden = true;
  ed.versions.forEach(v => v.img.close());
  edReset();
  ed.work.width = ed.orig.width = ed.maskC.width = ed.tmp2.width = ed.layer.width = 1;
}

// ---- derived state: everything is folded out of `acts`, so undo is a pop
function edPoints() {
  let pts = ed.basePoints.map(p => [...p]);
  for (const a of ed.acts) {
    if (a.t === 'click') pts.push(a.p);
    else if (a.t === 'sel' || a.t === 'clear') pts = [];
  }
  return pts;
}
function edVersion() {
  let v = 0;
  for (const a of ed.acts) if (a.t === 'version') v = a.to;
  return Math.min(v, ed.versions.length - 1);
}
function edOps() {   // hand-made pixel ops: they are what makes a result "edited"
  return ed.acts.filter(a => a.t === 'stroke' || a.t === 'sel').length;
}
function edSaveable() {   // a click alone changes no pixel, so it is nothing to save
  return ed.acts.some(a => a.t === 'stroke' || a.t === 'sel' || a.t === 'version');
}
function edIdle() {
  return ed.tool === 'brush' ? 'Paint over what to bring back or erase'
                             : 'Click the object you want to keep';
}
function paintStroke(ctx, s) {
  stamp(ctx, s.pts[0][0], s.pts[0][1], s);
  for (let i = 1; i < s.pts.length; i++) segment(ctx, s.pts[i - 1], s.pts[i], s);
}
function edReplay() {
  const ctx = ed.work.getContext('2d');
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, ed.W, ed.H);
  const v = ed.versions[edVersion()];
  if (v) ctx.drawImage(v.img, 0, 0, ed.W, ed.H);
  for (const a of ed.acts) {
    if (a.t === 'stroke') paintStroke(ctx, a);
    else if (a.t === 'sel' && a.mode === 'erase') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.drawImage(a.mask, 0, 0, ed.W, ed.H);
    } else if (a.t === 'sel') {
      // restore: the original seen through the mask, painted over the result
      const t = ed.tmp2; t.width = ed.W; t.height = ed.H;      // resizing clears it
      const tc = t.getContext('2d');
      tc.drawImage(a.mask, 0, 0, ed.W, ed.H);
      tc.globalCompositeOperation = 'source-in';
      tc.drawImage(ed.orig, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(t, 0, 0);
    }
    ctx.globalCompositeOperation = 'source-over';
  }
  if (ed.cur) paintStroke(ctx, ed.cur);   // a Generate landed mid-stroke: keep the stroke on top
  edButtons(); edVersions();
}
function pushAct(a) {
  ed.acts.push(a); ed.undone = [];
  // a stroke is already on the canvas, only the pixel ops below need a replay
  if (a.t === 'sel' || a.t === 'version') edReplay(); else edButtons();
  if (a.t !== 'stroke' && a.t !== 'version') { drawMask(); refreshMask(); }   // the dot at once
}
// undo pops from acts onto undone, redo the other way round; the popped act
// says what to refresh
function edStep(from, to, redo) {
  if (ed.cur || !from.length) return;          // not in the middle of a stroke
  const a = from.pop(); to.push(a);
  if (a.t !== 'click' && a.t !== 'clear') edReplay();
  if (a.t === 'sel') {                         // no refetch: the act kept its mask
    ed.seq++; ed.pending = false; ed.mask = redo ? null : a.mask;
    drawMask(); edButtons();
  } else if (a.t === 'click' || a.t === 'clear') { drawMask(); refreshMask(); }
}
const edUndo = () => edStep(ed.acts, ed.undone, false);
const edRedo = () => edStep(ed.undone, ed.acts, true);
function edButtons() {
  ed.undoB.disabled = ed.resetB.disabled = !ed.acts.length;
  ed.redoB.disabled = !ed.undone.length;
  ed.clearB.disabled = !edPoints().length;
  ed.selErase.disabled = ed.selRestore.disabled = !ed.mask || ed.pending;
  ed.done.disabled = !edSaveable() || !!ed.job || ed.saving;
  ed.gen.textContent = ed.job ? 'Cancel' : 'Generate';
}
function edVersions() {
  const cur = edVersion();
  if (ed.versEl.childElementCount !== ed.versions.length) {
    ed.versEl.innerHTML = '';
    ed.versions.forEach((v, i) => {
      const c = document.createElement('canvas');
      c.className = 'checkers'; c.width = c.height = 64;
      const s = Math.min(64 / v.img.width, 64 / v.img.height);
      const w = v.img.width * s, h = v.img.height * s;
      c.getContext('2d').drawImage(v.img, (64 - w) / 2, (64 - h) / 2, w, h);
      const b = document.createElement('button');
      b.className = 'ghost ed-ver';
      b.appendChild(c);
      b.appendChild(document.createTextNode(v.label));
      b.onclick = () => { if (i !== edVersion()) pushAct({ t: 'version', to: i }); };
      ed.versEl.appendChild(b);
    });
  }
  [...ed.versEl.children].forEach((b, i) => b.setAttribute('aria-pressed', i === cur));
}

// ---- brush: Restore stamps the original's pixels back through a soft
// circle, Erase cuts alpha with destination-out. Strokes are kept as data and
// replayed, so no pixel snapshots pile up.
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
  const t = ed.tmp;
  if (t.width < n || t.height < n) t.width = t.height = n;     // grows, never shrinks
  const tc = t.getContext('2d');
  tc.globalCompositeOperation = 'source-over'; tc.clearRect(0, 0, n, n);
  tc.fillStyle = brushGrad(tc, x - x0, y - y0, r, s.soft);
  tc.beginPath(); tc.arc(x - x0, y - y0, r, 0, 7); tc.fill();
  tc.globalCompositeOperation = 'source-in';
  tc.drawImage(ed.orig, x0, y0, n, n, 0, 0, n, n);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(t, 0, 0, n, n, x0, y0, n, n);
}
function segment(ctx, a, b, s) {
  const step = Math.max(1, s.size / 6);
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]), k = Math.max(1, Math.ceil(d / step));
  for (let i = 1; i <= k; i++)
    stamp(ctx, a[0] + (b[0] - a[0]) * i / k, a[1] + (b[1] - a[1]) * i / k, s);
}
function drawRing() {
  const c = ed.ring, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!ed.mouse || ed.tool !== 'brush') return;
  const r = ed.sizeEl.value * ed.zoom / 2;
  ctx.beginPath(); ctx.arc(ed.mouse[0], ed.mouse[1], r, 0, 7);
  ctx.lineWidth = 1.5; ctx.strokeStyle = ed.mode === 'erase' ? '#ff5252' : '#3ddc84'; ctx.stroke();
  ctx.beginPath(); ctx.arc(ed.mouse[0], ed.mouse[1], r, 0, 7);
  ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = .5; ctx.stroke();
}

// ---- SAM mask: preview from the server, hardened, drawn over everything
// the preview's soft alpha would leave half-transparent ghosts behind a
// destination-out, so remap it to a hard edge with a small feather
function hardenMask(bmp) {
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height), px = d.data;
  for (let i = 3; i < px.length; i += 4) {
    const t = Math.min(1, Math.max(0, (px[i] - 104) / 48));     // smoothstep around 128 ± 24
    px[i] = Math.round(255 * t * t * (3 - 2 * t));
  }
  ctx.putImageData(d, 0, 0);
  return c;
}
async function refreshMask() {
  const seq = ++ed.seq, open = ed.open, pts = edPoints();
  if (!pts.length) {
    ed.mask = null; ed.pending = false; drawMask(); edButtons();
    if (!ed.job) ed.msg.textContent = edIdle();
    return;
  }
  ed.pending = true; edButtons();          // no selection op on a mask about to change
  if (!ed.job) ed.msg.textContent = ed.mask ? 'Selecting…' : 'Reading the image…';
  const body = new FormData();
  body.append('source', ed.src.id); body.append('points', JSON.stringify(pts));
  try {
    await ensureAwake();
    const res = await fetch('/api/sam', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const bmp = await createImageBitmap(await res.blob());
    if (seq !== ed.seq || open !== ed.open) { bmp.close(); return; }   // a newer click already answered
    ed.mask = hardenMask(bmp); bmp.close();
    if (!ed.job) ed.msg.textContent = 'Bright with green outline = selected · apply it here or Generate';
  } catch (e) {
    if (seq !== ed.seq || open !== ed.open) return;
    ed.msg.textContent = 'Failed: ' + e.message;
  }
  ed.pending = false;
  drawMask(); edButtons();
}
// selected = full brightness with a green outline, the rest dimmed hard;
// built once per mask, zoom and clicks only redraw the dots on top
function buildMaskLayer() {
  const w = ed.maskC.width, h = ed.maskC.height;
  const L = ed.layer, lc = L.getContext('2d');
  L.width = w; L.height = h;                                    // resizing clears it
  lc.fillStyle = 'rgba(0,0,0,.6)'; lc.fillRect(0, 0, w, h);
  lc.globalCompositeOperation = 'destination-out';
  lc.drawImage(ed.mask, 0, 0, w, h);
  // outline: the mask in green, grown by d px, minus the mask itself
  const d = Math.max(2, Math.round(w / 500));
  const shape = document.createElement('canvas'); shape.width = w; shape.height = h;
  const sc = shape.getContext('2d');
  sc.drawImage(ed.mask, 0, 0, w, h);
  sc.globalCompositeOperation = 'source-in'; sc.fillStyle = '#3ddc84'; sc.fillRect(0, 0, w, h);
  const ring = document.createElement('canvas'); ring.width = w; ring.height = h;
  const rc = ring.getContext('2d');
  for (const [dx, dy] of [[d, 0], [-d, 0], [0, d], [0, -d], [d, d], [-d, -d], [d, -d], [-d, d]])
    rc.drawImage(shape, dx, dy);
  rc.globalCompositeOperation = 'destination-out'; rc.drawImage(shape, 0, 0);
  lc.globalCompositeOperation = 'source-over';
  lc.drawImage(ring, 0, 0);
  ed.layerFor = ed.mask;
}
function drawMask() {
  const c = ed.maskC, ctx = c.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, c.width, c.height);
  if (ed.mask) {
    if (ed.layerFor !== ed.mask) buildMaskLayer();
    ctx.drawImage(ed.layer, 0, 0);
  }
  // dots sit on the stretched canvas, so their radius follows how small it is drawn
  const sx = c.width / (ed.W || 1), rad = 6 * c.width / (ed.wrap.clientWidth || c.width);
  for (const [x, y, l] of edPoints()) {
    ctx.beginPath(); ctx.arc(x * sx, y * sx, rad, 0, 7);
    ctx.fillStyle = l ? '#3ddc84' : '#ff5252'; ctx.fill();
    ctx.lineWidth = rad / 3; ctx.strokeStyle = '#fff'; ctx.stroke();
  }
}

// ---- tools, view, zoom
function press(cls, key, v) {   // one pressed button per segmented group
  document.querySelectorAll('.' + cls).forEach(b => b.setAttribute('aria-pressed', b.dataset[key] == v));
}
function setTool(t) {
  ed.tool = t;
  press('ed-tool', 'tool', t);
  document.getElementById('ed-pick').hidden = t !== 'pick';
  document.getElementById('ed-brush').hidden = t !== 'brush';
  ed.el.classList.toggle('brush', t === 'brush');
  if (!ed.job && !ed.mask) ed.msg.textContent = edIdle();
  drawRing(); edHint();
}
function setLabel(l) { ed.label = l; press('ed-label', 'label', l); }
function setBrushMode(m) { ed.mode = m; press('ed-mode', 'mode', m); drawRing(); }
function setView(v) {
  ed.view = v;
  press('ed-view', 'view', v);
  ed.orig.hidden = v === 'result';
  ed.work.hidden = v === 'orig';
  ed.handle.hidden = v !== 'compare';
  setCompare(ed.compare);
}
function setCompare(p) {   // original on the left of the handle, result on the right
  ed.compare = Math.min(100, Math.max(0, p));
  ed.work.style.clipPath = ed.view === 'compare' ? `inset(0 0 0 ${ed.compare}%)` : '';
  ed.handle.style.left = ed.compare + '%';
}
function setBd(v) { ed.bd = v; ed.work.className = 'checkers ' + v; press('ed-bd', 'bd', v); }
function setZoom(z) {
  if (!ed.W) return;
  ed.fit = Math.min(1, (ed.stage.clientWidth - 8) / ed.W, (ed.stage.clientHeight - 8) / ed.H);
  const steps = [ed.fit, 1, 2, 4].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
  if (z === 'fit') ed.zoom = ed.fit;
  else if (z === 'in') ed.zoom = steps.find(s => s > ed.zoom + 1e-6) || ed.zoom;
  else if (z === 'out') ed.zoom = [...steps].reverse().find(s => s < ed.zoom - 1e-6) || ed.zoom;
  ed.wrap.style.width = (ed.W * ed.zoom) + 'px';
  ed.wrap.style.height = (ed.H * ed.zoom) + 'px';
  requestAnimationFrame(() => {
    const r = ed.stage.getBoundingClientRect();
    ed.ring.width = r.width; ed.ring.height = r.height;
    ed.ring.style.width = r.width + 'px'; ed.ring.style.height = r.height + 'px';
    drawRing(); drawMask();
  });
}
function edHint() {
  const s = MODELS.sam || {};
  const tool = ed.tool === 'brush'
    ? 'drag to paint · [ ] size · X swaps Restore/Erase'
    : (s.downloaded ? '' : `first use downloads SAM 2 (~${s.size_mb} MB) · `) +
      'click = keep · ⌥-click or right-click = exclude · Erase/Restore selection applies it ' +
      'here · Generate reruns the model with the clicks';
  ed.hint.textContent = tool + ' · ⌘Z undo · ⌘⇧Z redo · Enter = Done · Esc closes';
}

// ---- pointer work. Every layer is pointer-events:none, so the wrap is the
// target in all view modes, the clipped Compare one included.
function edPos(e) {
  const r = ed.wrap.getBoundingClientRect();
  return [(e.clientX - r.left) / ed.zoom, (e.clientY - r.top) / ed.zoom];
}
function edClick(e, label) {
  const [x, y] = edPos(e);
  pushAct({ t: 'click', p: [Math.round(x), Math.round(y), label] });
}
ed.wrap.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !ed.W || ed.saving) return;
  e.preventDefault();
  if (ed.tool === 'pick') { edClick(e, e.altKey ? 0 : ed.label); return; }
  if (ed.view === 'orig') setView('result');       // paint on what the stroke changes
  ed.wrap.setPointerCapture(e.pointerId);
  const p = edPos(e);
  ed.cur = { t: 'stroke', mode: ed.mode, size: +ed.sizeEl.value, soft: ed.softEl.value / 100, pts: [p] };
  stamp(ed.work.getContext('2d'), p[0], p[1], ed.cur);
});
ed.wrap.addEventListener('pointermove', e => {
  if (!ed.cur) return;
  const p = edPos(e), last = ed.cur.pts[ed.cur.pts.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 1) return;
  segment(ed.work.getContext('2d'), last, p, ed.cur);
  ed.cur.pts.push(p);
});
function endStroke() {
  if (!ed.cur) return;
  const s = ed.cur; ed.cur = null;
  pushAct(s);
}
ed.wrap.addEventListener('pointerup', endStroke);
ed.wrap.addEventListener('pointercancel', endStroke);
ed.wrap.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (ed.tool === 'pick' && ed.W && !ed.saving) edClick(e, 0);
});
ed.stage.addEventListener('pointermove', e => {
  const r = ed.stage.getBoundingClientRect();
  ed.mouse = [e.clientX - r.left, e.clientY - r.top]; drawRing();
});
ed.stage.addEventListener('pointerleave', () => { ed.mouse = null; drawRing(); });
ed.handle.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); });
ed.handle.addEventListener('pointerdown', e => {
  e.preventDefault(); e.stopPropagation();          // dragging the handle is not a stroke
  ed.handle.setPointerCapture(e.pointerId);
  const move = ev => {
    const r = ed.wrap.getBoundingClientRect();
    setCompare((ev.clientX - r.left) / r.width * 100);
  };
  const up = () => { ed.handle.onpointermove = ed.handle.onpointerup = ed.handle.onpointercancel = null; };
  ed.handle.onpointermove = move; ed.handle.onpointerup = ed.handle.onpointercancel = up;
});

// ---- panel wiring
for (const [cls, key, fn] of [['ed-tool', 'tool', setTool], ['ed-label', 'label', l => setLabel(+l)],
                              ['ed-mode', 'mode', setBrushMode], ['ed-view', 'view', setView],
                              ['ed-bd', 'bd', setBd]])
  document.querySelectorAll('.' + cls).forEach(b => b.addEventListener('click', () => fn(b.dataset[key])));
document.getElementById('ed-zoomin').addEventListener('click', () => setZoom('in'));
document.getElementById('ed-zoomout').addEventListener('click', () => setZoom('out'));
document.getElementById('ed-zoomfit').addEventListener('click', () => setZoom('fit'));
ed.sizeEl.addEventListener('input', drawRing);
ed.undoB.addEventListener('click', edUndo);
ed.redoB.addEventListener('click', edRedo);
ed.resetB.addEventListener('click', () => {
  if (ed.cur || ed.saving) return;
  ed.acts = []; ed.undone = []; edReplay(); drawMask(); refreshMask();
});
ed.clearB.addEventListener('click', () => { if (edPoints().length) pushAct({ t: 'clear' }); });
// applying a selection here costs no model run, but its edges are as coarse
// as the SAM preview; the hint sends fine work to Generate
ed.selErase.addEventListener('click', () => selApply('erase'));
ed.selRestore.addEventListener('click', () => selApply('restore'));
function selApply(mode) {
  if (!ed.mask || ed.pending || ed.saving) return;
  pushAct({ t: 'sel', mode, mask: ed.mask });
}
document.getElementById('ed-close').addEventListener('click', () => closeEditor());

// Generate: a normal /api/cutout run, but draft=1 so it stays in the editor
// instead of becoming a card, and always transparent (Done flattens)
ed.gen.addEventListener('click', async () => {
  if (ed.job) { ed.ctrl.abort(); cancelJob(ed.job); return; }   // running: the button reads Cancel
  if (!ed.W || ed.saving) return;
  const model = ed.modelEl.value, tta = ed.ttaEl.checked, pts = edPoints();
  const open = ed.open, job = crypto.randomUUID();
  ed.job = job; ed.ctrl = new AbortController();
  const body = new FormData();
  body.append('source', ed.src.id);
  body.append('bg', '');
  body.append('model', model);
  body.append('tta', tta ? '1' : '');
  body.append('job', job);
  body.append('draft', '1');
  body.append('points', JSON.stringify(pts));
  edButtons();
  try {
    const res = await submit('/api/cutout', body, job, ed.ctrl, ed.msg, model);
    const blob = await res.blob();
    const img = await createImageBitmap(blob);
    if (open !== ed.open) { img.close(); return; }     // closed while it ran
    const n = pts.length;
    ed.versions.push({ img, blob, points: pts, model, tta,
      label: LABEL[model] + (n ? ` · ${n} click${n > 1 ? 's' : ''}` : '') });
    pushAct({ t: 'version', to: ed.versions.length - 1 });
    ed.msg.textContent = 'New version · pick an older one in Versions to go back';
  } catch (err) {
    if (open !== ed.open) return;
    ed.msg.textContent = err.name === 'AbortError' ? 'Generate cancelled' : 'Failed: ' + err.message;
  } finally {
    if (open === ed.open) { ed.job = null; ed.ctrl = null; edButtons(); }
  }
});

// Done: the composed pixels become a card, tagged with the model, clicks and
// extra pass of the version they came from
ed.done.addEventListener('click', async () => {
  if (ed.done.disabled || ed.cur) return;
  const open = ed.open, v = ed.versions[edVersion()], ops = edOps();
  ed.saving = true; edButtons(); ed.msg.textContent = 'Saving…';
  // untouched by hand: the server's own PNG, not a canvas round trip
  const blob = ops ? await new Promise(r => ed.work.toBlob(r, 'image/png')) : v.blob;
  // the editor stays open until the save succeeded: a failed upload must not
  // throw the work away
  const ok = await run(ed.src, { ...ed.opts, model: v.model, tta: v.tta, points: v.points,
                                 edited: ops > 0 },
                       ed.anchor, { url: '/api/edit', blob, strokes: ops });
  if (open !== ed.open) return;        // closed meanwhile: the card stands on its own
  ed.saving = false;
  if (ok) closeEditor(true);
  else { ed.msg.textContent = 'Saving failed (the card behind says why); your changes are kept'; edButtons(); }
});

window.addEventListener('resize', () => { if (!ed.el.hidden) setZoom(ed.zoom === ed.fit ? 'fit' : 'same'); });
window.addEventListener('keydown', e => {
  if (ed.el.hidden) return;
  // a slider or checkbox keeps focus after a drag; only text fields and the
  // model menu really own the letter keys
  const typing = e.target instanceof Element &&
    e.target.matches('select, textarea, input:not([type=range]):not([type=checkbox])');
  if (e.key === 'Escape') closeEditor();
  else if (e.key === 'Enter') { e.preventDefault(); if (!ed.done.disabled) ed.done.click(); }
  else if (e.key.toLowerCase() === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); (e.shiftKey ? edRedo : edUndo)(); }
  else if (typing) return;
  else if (e.key === '[') { ed.sizeEl.value = Math.max(2, ed.sizeEl.value / 1.25); drawRing(); }
  else if (e.key === ']') { ed.sizeEl.value = Math.min(ed.sizeEl.max, ed.sizeEl.value * 1.25); drawRing(); }
  else if (e.key === '1') setView('orig');
  else if (e.key === '2') setView('result');
  else if (e.key === '3') setView('compare');
  else if (e.key === 'p' || e.key === 'P') setTool('pick');
  else if (e.key === 'b' || e.key === 'B') setTool('brush');
  else if (e.key === 'r' || e.key === 'R') setBrushMode('restore');
  else if (e.key === 'e' || e.key === 'E') setBrushMode('erase');
  else if (e.key === 'x' || e.key === 'X') setBrushMode(ed.mode === 'erase' ? 'restore' : 'erase');
});

(async () => {
  try { await loadModels(); }
  catch (e) { hint.textContent = 'model list failed: ' + e.message; }
  await restoreSettings();
  refreshStatus();
  statusTimer = setInterval(refreshStatus, 1500);
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
            "ram_gb": round(rmbg.total_ram_gb()), "jobs": dict(JOBS)}


@app.get("/api/settings")
def api_settings():
    return load_settings()


@app.post("/api/settings")
async def api_settings_set(request: Request):
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(400, "settings must be a JSON object")
    s = load_settings()
    if "model" in data:
        if data["model"] not in rmbg.MODELS:
            raise HTTPException(400, f"unknown model {data['model']!r}")
        s["model"] = data["model"]
    if "bg" in data:
        parse_bg(data["bg"])
        s["bg"] = str(data["bg"] or "")
    if "tta" in data:
        s["tta"] = bool(data["tta"])
    save_settings(s)
    return s


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
    image = rmbg.open_rgb(os.path.join(history_dir(source), "orig" + meta["ext"]))
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
        rmbg.open_rgb(path).save(preview, quality=88)
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
def api_edit(source: str = Form(""), image: UploadFile = File(...), strokes: int = Form(0),
             model: str = Form(""), tta: str = Form(""), points: str = Form("")):
    """A cutout composed in the browser's editor (brush strokes, selection
    ops, a generated version, or all three): store it as a new history entry
    next to its source (same original and background). model/tta/points say
    what really produced these pixels; they are labels here, nothing runs, so
    a model name this version does not know (an old entry) is kept as the
    source's. `strokes` counts the pixel ops by hand, so `edited` marks only
    what a person painted."""
    meta = history_meta(source)
    if model not in rmbg.MODELS:
        model = meta["model"]
    pts = parse_points(points)
    bg = str(meta.get("bg") or "")
    background = parse_bg(bg)
    rgba = Image.open(io.BytesIO(image.file.read())).convert("RGBA")
    with open(os.path.join(history_dir(source), "orig" + meta["ext"]), "rb") as f:
        raw = f.read()
    out = rmbg.flatten(rgba, background) if background else rgba
    png = png_bytes(out)
    headers = {}
    if STATE["history_days"] > 0:
        headers["X-Id"] = history_save(raw, meta["name"], png, {
            "model": model, "bg": bg, "tta": bool(tta),
            "points": pts, "edited": strokes > 0, "strokes": strokes,
            "seconds": 0, "width": rgba.width, "height": rgba.height},
            cut=png_bytes(rgba) if bg else None)
    return Response(png, media_type="image/png", headers=headers)


@app.delete("/api/history")
def api_history_clear():
    for m in history_list():
        shutil.rmtree(os.path.join(HISTORY_DIR, m["id"]), ignore_errors=True)
    return {"deleted": "all"}


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
    use_settings: str = Form(""),  # model/bg/tta from the page's settings (Quick Action)
    draft: str = Form(""),         # editor preview: send the PNG back, write no history
):
    # sync endpoint on purpose: FastAPI runs it in a worker thread, so the
    # event loop keeps answering /api/status while the GPU is busy.
    if use_settings:
        s = load_settings()
        model = s.get("model") if s.get("model") in rmbg.MODELS else model
        bg = str(s.get("bg") or "")
        tta = "1" if s.get("tta") else ""
    if model not in rmbg.MODELS:
        return Response(f"unknown model {model!r}", status_code=400)
    background = parse_bg(bg)      # a bad colour fails here, not after the GPU run
    if image is not None:
        raw, name = image.file.read(), image.filename or "image"
    elif source:
        meta = history_meta(source)
        name = meta["name"]
        with open(os.path.join(history_dir(source), "orig" + meta["ext"]), "rb") as f:
            raw = f.read()
    else:
        return Response("no image", status_code=400)
    try:
        src = rmbg.open_rgb(io.BytesIO(raw))
    except Exception:
        return Response("not an image", status_code=400)
    pts = parse_points(points)

    t0 = time.time()
    try:
        JOBS[job] = "queued"
        with GPU:
            if job in CANCELLED:
                return Response("cancelled", status_code=499)
            touch(model)
            if model not in loaded_names():
                JOBS[job] = "load" if repo_downloaded(rmbg.MODELS[model][0])[0] else "download"
            net, size, device, half = rmbg.load_model(
                model, rmbg.pick_device(STATE["device"]),
                half=False if STATE["fp32"] else None)
            size = STATE["size"] or size
            if pts:
                touch(sam.NAME)
                JOBS[job] = "run" if sam.loaded() else "sam"
                try:
                    rgba, _ = sam.cutout_picked(src, pts, net, size, device, half=half,
                                                tta=bool(tta), key=source or None)
                except ValueError as e:
                    return Response(str(e), status_code=422)
                touch(sam.NAME)
            else:
                JOBS[job] = "run"
                rgba, _ = rmbg.cutout(src, net, size, device, half=half, tta=bool(tta))
            touch(model)
    finally:
        JOBS.pop(job, None)

    out = rmbg.flatten(rgba, background) if background else rgba

    png = png_bytes(out)
    headers = {"X-Seconds": f"{time.time() - t0:.2f}"}
    # a draft is one of the editor's tries, not a result: no entry, no X-Id, so
    # it never becomes a card (with --history-days 0 nothing is written anyway)
    if STATE["history_days"] > 0 and job not in CANCELLED and not draft:
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
                    help="keep results on disk this long, 0 = write nothing new")
    ap.add_argument("--no-warmup", action="store_true",
                    help="do not load the model at startup, wait for the first image")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    STATE.update(device=args.device, fp32=args.fp32, idle=args.idle, size=args.size,
                 sleep_after=args.sleep_after, history_days=args.history_days,
                 last_work=time.time(), host=args.host)
    history_purge()
    if not args.no_warmup:
        print(f"warming up {rmbg.MODELS[args.model][0]} ...", flush=True)
        rmbg.load_model(args.model, rmbg.pick_device(args.device),
                        half=False if args.fp32 else None)
        touch(args.model)
    for m in loaded_models():
        print(f"{m['model']}: {m['precision']} at {m['size']}px on {m['device']}, "
              f"{rmbg.total_ram_gb():.0f} GB RAM, idle unload after {args.idle}s")
    for step, every in ((idle_step, 2), (sleep_step, 15), (purge_step, 3600)):
        threading.Thread(target=forever, args=(step, every), daemon=True).start()
    print(f"\n  open  http://{args.host}:{args.port}\n", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
