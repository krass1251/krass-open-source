"""
Drag-and-drop web UI for rmbg.py. A local remove.bg, on localhost.

    ./run-ui.sh            # or: .venv/bin/python server.py

Then open http://127.0.0.1:8777 and drop photos on the page. Nothing leaves
the machine: the browser talks to 127.0.0.1 and images are never written to
disk by the server.

Memory: a model loads on the first image that asks for it and is dropped
again after --idle seconds without work for that model (default 60). Several
models can sit in memory at once (weights are ~0.5 GB each; the 4-16 GB peak
is activations, released after every image), so switching models to compare
does not reload anything. A batch of photos in a row runs on the warm model.
Loading costs ~1-4 s, unloading ~0.3 s. The x on a model chip (POST
/api/unload with model=...) drops one model, "Free all" drops every one.

The "Quit" button (POST /api/quit) stops the server, for when it was started
from "Remove Background.app" (see make-app.sh) and there is no terminal.
"""

import argparse
import io
import os
import threading
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image

import rmbg

app = FastAPI(title="remove-bg (local)")

STATE = {"device": "auto", "fp32": False, "idle": 60, "last_used": {},
         "size": None}
GPU = threading.Lock()   # one inference at a time; also guards load/unload


def touch(model):
    STATE["last_used"][model] = time.time()


def loaded_names():
    return {key[0] for key in list(rmbg._LOADED)}


def idle_for(model):
    return time.time() - STATE["last_used"].get(model, 0)


def loaded_models():
    """What is in memory right now, for /api/status: one entry per model."""
    out = []
    for (model, device, half), (_, size, _, _) in list(rmbg._LOADED.items()):
        left = None
        if STATE["idle"] > 0:
            left = max(0, int(STATE["idle"] - idle_for(model)))
        out.append({"model": model, "device": device,
                    "precision": "fp16" if half else "fp32",
                    "size": STATE["size"] or size, "unload_in": left})
    return out


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
                    rmbg.unload_model(model)
                    print(f"{model} unloaded after idle", flush=True)


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
  .cards { display: grid; gap: 14px; margin-top: 18px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px; display: grid; grid-template-columns: 1fr 1fr auto; gap: 14px;
    align-items: center;
  }
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
  .actions { display: flex; flex-direction: column; gap: 8px; }
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
    display: flex; gap: 12px; align-items: center; margin-top: 10px;
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
  .status { flex-wrap: wrap; }
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
  .actions select { font-size: 12px; max-width: 170px; }
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
    <label>Model
      <select id="model">
        <option value="hr-matting">HR matting (best)</option>
        <option value="hr">HR (crisp edges)</option>
        <option value="matting">matting 1024 (fast)</option>
        <option value="general">general 1024</option>
        <option value="portrait">portrait</option>
        <option value="lite">lite (fastest)</option>
      </select>
    </label>
    <label><input type="checkbox" id="tta"> extra pass (slower, cleaner)</label>
  </div>

  <div class="status">
    <span id="stat">checking…</span>
    <div class="chips" id="chips"></div>
    <button class="ghost" id="free" disabled>Free all</button>
    <button class="ghost" id="quit">Quit</button>
  </div>

  <div class="cards" id="cards"></div>
</main>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const cards = document.getElementById('cards');
let bg = '';

document.getElementById('sw').addEventListener('click', e => {
  const b = e.target.closest('.sw'); if (!b) return;
  document.querySelectorAll('.sw').forEach(s => s.setAttribute('aria-pressed', s === b));
  bg = b.dataset.bg;
});
document.getElementById('pick').addEventListener('input', e => {
  bg = e.target.value;
  document.querySelectorAll('.sw').forEach(s => s.setAttribute('aria-pressed', false));
});

drop.addEventListener('click', () => file.click());
file.addEventListener('change', () => handle(file.files));
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

const stat = document.getElementById('stat');
const chips = document.getElementById('chips');
const free = document.getElementById('free');
const modelSel = document.getElementById('model');
const LABEL = Object.fromEntries([...modelSel.options].map(o => [o.value, o.textContent]));
let busy = 0;
const working = {};   // model -> images in flight

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function refreshStatus() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (e) { stat.textContent = 'server not reachable'; return; }
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
let statusTimer = setInterval(refreshStatus, 3000);
refreshStatus();
document.getElementById('quit').addEventListener('click', async () => {
  if (busy && !confirm('Images are still processing. Quit anyway?')) return;
  clearInterval(statusTimer);
  try { await fetch('/api/quit', { method: 'POST' }); } catch (e) {}
  document.querySelector('main').innerHTML =
    '<p class="sub">Server stopped. You can close this tab; open Remove Background again to start it.</p>';
});

function handle(files) {
  const opts = { model: modelSel.value, bg, tta: document.getElementById('tta').checked };
  [...files].filter(f => f.type.startsWith('image/')).forEach(f => run(f, opts));
}

// One card per (image, model) run. Redo runs the same image with the same
// background and extra-pass setting through the model picked on the card, as
// a new card right above it, so the two results sit next to each other.
async function run(f, opts, anchor) {
  const card = document.createElement('div');
  card.className = 'card';
  const options = [...modelSel.options].map(o =>
    `<option value="${o.value}"${o.value === opts.model ? ' selected' : ''}>` +
    `${esc(o.textContent)}</option>`).join('');
  card.innerHTML = `
    <div class="shot"><img></div>
    <div class="shot checker"><div class="spin"></div></div>
    <div class="actions">
      <button class="go" disabled>Save</button>
      <select class="redo-model" title="model for Redo">${options}</select>
      <button class="ghost redo">Redo with this model</button>
    </div>
    <div class="meta">
      <span>${esc(f.name)} · <span class="tag">${esc(LABEL[opts.model] || opts.model)}</span>` +
      `${opts.tta ? ' · extra pass' : ''}</span>
      <span class="t">working…</span>
    </div>`;
  if (anchor) anchor.before(card); else cards.prepend(card);
  const [before, after] = card.querySelectorAll('.shot');
  before.querySelector('img').src = URL.createObjectURL(f);
  card.querySelector('.redo').onclick = () =>
    run(f, { ...opts, model: card.querySelector('.redo-model').value }, card);

  const body = new FormData();
  body.append('image', f);
  body.append('bg', opts.bg);
  body.append('model', opts.model);
  body.append('tta', opts.tta ? '1' : '');

  const t0 = performance.now();
  busy++; working[opts.model] = (working[opts.model] || 0) + 1; refreshStatus();
  try {
    const res = await fetch('/api/cutout', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    after.innerHTML = '<img src="' + url + '">';
    if (opts.bg) after.classList.remove('checker');
    const name = f.name.replace(/\.[^.]+$/, '') + `.${opts.model}.cutout.png`;
    const btn = card.querySelector('.go');
    btn.disabled = false;
    btn.onclick = () => {
      const a = document.createElement('a');
      a.href = url; a.download = name; a.click();
    };
    card.querySelector('.t').textContent =
      ((performance.now() - t0) / 1000).toFixed(1) + ' s';
  } catch (err) {
    after.innerHTML = '<div class="err"></div>';
    after.querySelector('.err').textContent = err.message;
    card.querySelector('.t').textContent = 'failed';
  } finally {
    busy--; working[opts.model]--; refreshStatus();
  }
}
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


@app.post("/api/unload")
def api_unload(model: str = Form("")):
    """Drop one model (form field `model`) or, without it, every model."""
    with GPU:
        if model:
            rmbg.unload_model(model)
        else:
            rmbg.unload_models()
    return {"loaded": loaded_models()}


@app.post("/api/quit")
def api_quit():
    # exit a moment later so this response still reaches the browser
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"quit": True}


@app.post("/api/cutout")
def api_cutout(
    image: UploadFile = File(...),
    bg: str = Form(""),
    model: str = Form("hr-matting"),
    tta: str = Form(""),
):
    # sync endpoint on purpose: FastAPI runs it in a worker thread, so the
    # event loop keeps answering /api/status while the GPU is busy.
    if model not in rmbg.MODELS:
        return Response(f"unknown model {model!r}", status_code=400)
    raw = image.file.read()
    src = Image.open(io.BytesIO(raw)).convert("RGB")

    with GPU:
        touch(model)
        net, size, device, half = rmbg.load_model(
            model, rmbg.pick_device(STATE["device"]),
            half=False if STATE["fp32"] else None)
        size = STATE["size"] or size
        rgba, _ = rmbg.cutout(src, net, size, device, half=half, tta=bool(tta))
        touch(model)

    if bg:
        out = rmbg.flatten(rgba, rmbg.parse_background(bg))
    else:
        out = rgba

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("-m", "--model", default="hr-matting", choices=list(rmbg.MODELS),
                    help="checkpoint to warm up at startup")
    ap.add_argument("-s", "--size", type=int,
                    help="inference resolution (default follows RAM)")
    ap.add_argument("--idle", type=int, default=60,
                    help="unload the model after this many idle seconds, 0 = never")
    ap.add_argument("--no-warmup", action="store_true",
                    help="do not load the model at startup, wait for the first image")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    STATE.update(device=args.device, fp32=args.fp32, idle=args.idle, size=args.size)
    if not args.no_warmup:
        print(f"warming up {rmbg.MODELS[args.model][0]} ...", flush=True)
        rmbg.load_model(args.model, rmbg.pick_device(args.device),
                        half=False if args.fp32 else None)
        touch(args.model)
    for m in loaded_models():
        print(f"{m['model']}: {m['precision']} at {m['size']}px on {m['device']}, "
              f"{rmbg.total_ram_gb():.0f} GB RAM, idle unload after {args.idle}s")
    threading.Thread(target=idle_reaper, daemon=True).start()
    print(f"\n  open  http://{args.host}:{args.port}\n", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
