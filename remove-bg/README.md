# remove-bg

A local replacement for remove.bg. Same class of model, no upload, no credits,
no per-image limit.

```bash
./run-ui.sh                       # drag & drop UI on http://127.0.0.1:8777
.venv/bin/python rmbg.py cat.jpg  # or straight from the shell
```

The engine is [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (CAAI AIR'24),
the architecture the current crop of commercial background removers is built
on. The default checkpoint, `BiRefNet_HR-matting`, runs at 2048x2048, which is
what keeps hair, fur and semi-transparent edges intact. MIT licensed, so
commercial use is fine.

Weights (~900 MB) download once into `~/.cache/huggingface`. After that it
works offline.

## Moving it to another Mac

```bash
git clone <this repo> && cd krass-open-source/remove-bg
./setup.sh          # uv + Python 3.12 venv + packages + weights, ~5 min
./run-ui.sh
```

`setup.sh` is safe to re-run. It picks the inference resolution from the
machine's RAM automatically (see Memory below), so nothing needs tuning on a
16 GB M1/M2. To skip the 900 MB download on a machine without internet, copy
`~/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet_HR-matting` over.

## Web UI

```bash
./run-ui.sh
```

Open <http://127.0.0.1:8777>, then drop files on the page, click to pick them,
or just press ⌘V to paste from the clipboard. Pick a background colour or keep
transparency, hit **Save** or **Copy** (the PNG lands on the clipboard, ready
for Figma, Telegram, Keynote). The model is loaded once at startup and stays
in memory, so only the first image pays for the warm-up. **Cancel** on a card
drops a job that has not started yet; the one already on the GPU finishes in
a few seconds and is discarded.

The browser only ever talks to `127.0.0.1`; nothing leaves the machine.

Model, background and extra pass are remembered between visits. The line
under the toolbar says what the selected model is good for and whether its
weights are on disk yet (each is a ~430 MB download on first use, lite 170).

### Pick object: click on what to keep

BiRefNet decides by itself what the subject is, which is right most of the
time. When a photo has several things in it, **Pick object** on a card opens
the photo large: click the object to keep, ⌥-click (or right-click) parts to
drop; what stays is shown bright with a green outline, what goes is dimmed.
**Redo with selection** makes a new card with only
that object. Clicks are remembered with the result, so Redo with another
model keeps the same object and Pick object on that card starts from them.

Under the hood this is [SAM 2](https://github.com/facebookresearch/sam2)
(`sam2.1-hiera-small`, 39M parameters, Apache-2.0, ~180 MB download on first
use) picking the object and BiRefNet doing the edges: the object is cropped
out with a margin, BiRefNet runs on the crop, and its alpha is kept inside
the SAM region. If BiRefNet does not see the object at all, the SAM mask is
used as is (coarser edges, but the right object). SAM runs on the CPU, ~1 s
to read an image once and ~30 ms per click after that, and shows up in the
memory line as `sam2` with the same idle unload as the other models.

### Touch up: the brush

When the model got a detail wrong, **Touch up** on a card opens the cutout
large on a checkerboard (or white / black, to see the edges). **Restore**
paints the original's pixels back, **Erase** makes them transparent; size and
softness sliders, zoom to 1x/2x/4x with scrolling for hair, Undo / Redo /
Reset. **Apply changes** makes a new card above the source, tagged `brush`.
Keys: `[` `]` size, `X` swaps the tools, `⌘Z` / `⌘⇧Z`, Enter applies, Esc
closes. Everything happens in the browser on the full-resolution image; no
model runs. Results that were flattened onto a colour keep their transparent
cutout on disk (`cut.png`), so the brush works on those too.

### History

Every result is kept for 7 days in
`~/Library/Application Support/remove-bg/history` (original + cutout), so the
page shows the same cards after a reload or a server restart, and **Redo**
still works on them. **Delete** on a card removes it from disk at once.
`--history-days 30` keeps them longer, `--history-days 0` writes nothing to
disk. Note that big photos make big PNGs: check the folder size if disk space
is tight.

### As a Mac app, no terminal

```bash
./make-app.sh
```

Builds two things, both pointing at this folder (re-run after moving the
repo; safe to re-run):

* `~/Applications/Remove Background.app`. Open it from Launchpad or Spotlight,
  or drag it to the Dock: it starts the server in the background if it is not
  running (`serve.sh`) and opens the page in Chrome (default browser if Chrome
  is not installed).
* A Finder Quick Action **Remove Background**: right-click one or more photos
  → Quick Actions → Remove Background, and `<name>.cutout.png` appears next to
  each file (hr-matting, transparent). It goes through the same server, so
  the warm model is reused and the results show up in the web UI's history
  too. A notification reports how many were done.

After 10 minutes without an image the server goes to sleep: the ~1 GB torch
process swaps itself for `sleeper.py`, a ~15 MB stand-in on the same port
(`--sleep-after 30` to change, `0` to never sleep). Open tabs do not keep it
awake and do not go stale either: the page shows "server sleeping", and the
next photo dropped on it (or **Wake up**) brings the real server back in a
few seconds; so does opening the URL fresh, the app, or the Quick Action.
**Quit** stops it for good. Server output goes to `~/Library/Logs/remove-bg.log`.

### Memory

The status line under the toolbar lists every model in memory, with its
precision, resolution and when it frees itself. Each model unloads after 60 s
without work of its own (`--idle 120` to change, `--idle 0` to keep them
forever); the × on a model drops just that one, **Free all** drops every one.
Loading again costs about a second, so a batch of photos still runs on the
warm model.

Every result card says which model made it. To compare, pick another model on
the card and hit **Redo with this model**: the same photo, background and
extra-pass setting run again, and the new card lands right above the old one.
Saved files carry the model name, `photo.hr-matting.cutout.png`.

```
./run-ui.sh --idle 300        # keep the model 5 min after the last image
./run-ui.sh --no-warmup       # start instantly, load on the first image
./run-ui.sh -s 2048           # force a resolution
./run-ui.sh --history-days 0  # keep nothing on disk
./run-ui.sh --sleep-after 0   # never swap to the sleeper
```

## CLI

```bash
.venv/bin/python rmbg.py photo.jpg              # -> photo.cutout.png, transparent
.venv/bin/python rmbg.py ~/Desktop/shots/       # whole folder
.venv/bin/python rmbg.py photo.jpg --bg white   # flatten onto white
.venv/bin/python rmbg.py photo.jpg --jpg        # jpg on white, for marketplaces
.venv/bin/python rmbg.py photo.jpg --bg room.jpg --mask
```

Results land next to the input as `<name>.cutout.png`, unless `-o` says
otherwise.

| flag | what it does |
|---|---|
| `-o, --out` | output file (single input) or folder |
| `-m, --model` | checkpoint, see the table below (default `hr-matting`) |
| `-s, --size` | inference resolution, default follows RAM (see Memory) |
| `-b, --bg` | flatten onto `white`, `#e9e4dc`, `255,0,0` or a background image |
| `--jpg` | write jpg instead of png (implies white background) |
| `--mask` | also write `<name>.mask.png`, the raw alpha |
| `--tta` | average with the horizontal flip: ~2x slower, slightly cleaner |
| `--no-unmix` | skip foreground colour estimation (faster, can leave a halo) |
| `--fp32` | fp32 inference: 2x memory, 3-4x slower, same output. fp16 is the default on GPU |
| `--device` | `auto` \| `mps` \| `cuda` \| `cpu` |

## Models

| `--model` | resolution | when |
|---|---|---|
| `hr-matting` | 2048 | **default.** Best edges. Hair, fur, glass, smoke. |
| `hr` | 2048 | Same resolution, harder edges. Products, logos, packshots. |
| `matting` | 1024 | 4x faster, softer detail. Good enough for web-size images. |
| `portrait` | 1024 | Tuned for people. |
| `lite` | 1024 | Small backbone, the one to use without a GPU. |

Each one downloads on first use.

## Memory and speed

Measured on an M3 Max with `hr-matting`. "peak" is what the GPU allocator
grabs during one image; that is the number that has to fit in RAM.

| precision | resolution | peak | per image | vs 2048 fp32 |
|---|---|---|---|---|
| fp32 | 2048 | 34 GB | 6.9 s | — |
| fp16 | 2048 | 16 GB | 1.9 s | mean alpha diff 0.2/255 |
| fp16 | 1536 | 8 GB | 1.0 s | 0.2% of pixels differ by >8/255 |
| fp16 | 1280 | 6 GB | 0.8 s | 0.3% |
| fp16 | 1024 | 4 GB | 0.5 s | 0.4% |

So fp16 is the default (same output, a third of the time) and the resolution
is picked from RAM: 2048 on 32 GB+, 1536 on 16 GB, 1024 below. An M1/M2 with
16 GB runs 1536 fp16 in roughly 3–4 s per image. Override with `-s`.

Load ~1 s, unload ~0.3 s (warm disk cache). Model in memory: ~0.5 GB of
weights; the big number is the activations during an image, and they are
released after each one.

## Why the output beats a plain `rembg`

* inference at the model's native 2048px instead of 1024px, so thin hair
  survives the resize
* the alpha is scaled back to the original size with LANCZOS, not nearest
* `pymatting` estimates the true foreground colour, so a cutout taken off a
  dark background does not keep a grey rim when you drop it on white

## Install

Already done — `.venv/` next to this file has everything. To rebuild it:

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

The venv is pinned to 3.12 because the system Python here is 3.14, which torch
has no wheels for. It is fully independent of the system Python.
