"""
Remove the background from a photo, locally. A remove.bg replacement.

    python3 rmbg.py /path/to/cat.jpg
    python3 rmbg.py /path/to/folder --bg white

Writes <name>.cutout.png (RGBA, transparent background) next to the input.

Under the hood this is BiRefNet (CAAI AIR'24), the model family remove.bg-class
services are built on -- the HR matting checkpoint runs at 2048x2048, which is
what keeps hair, fur and semi-transparent edges alive. Everything is local:
weights are cached in ~/.cache/huggingface once, then it works offline.

Memory and precision (measured on an M3 Max, hr-matting):
  fp32 @ 2048px  34 GB peak, 6.9 s/image      fp16 @ 2048px  16 GB, 1.9 s
  fp16 @ 1536px   8 GB peak, 1.0 s/image      fp16 @ 1024px   4 GB, 0.5 s
fp16 output is indistinguishable from fp32 (mean alpha diff 0.2/255), so fp16
is the default on GPU. The default resolution follows the machine's RAM:
2048 with 32 GB+, 1536 with 16 GB, 1024 below that. Override with --size.

Quality tricks that are on by default:
  * inference at the model's native resolution, alpha resized back with LANCZOS
  * pymatting foreground estimation, so hair does not keep a dark halo of the
    old background when you put the cutout on a light backdrop
  * --tta averages the horizontal flip, a small but free accuracy gain

Needs: torch, torchvision, transformers, pillow, timm, einops, kornia, pymatting
(see README; the .venv next to this file already has them).
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image, ImageColor
from torchvision import transforms

try:  # iPhone photos are .heic, Pillow needs a plugin for those
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

# ------------------------------------------------------------------- models

# name -> (hugging face repo, native inference resolution)
MODELS = {
    "hr-matting": ("ZhengPeng7/BiRefNet_HR-matting", 2048),  # best edges, default
    "hr":         ("ZhengPeng7/BiRefNet_HR", 2048),          # crisp binary cutouts
    "matting":    ("ZhengPeng7/BiRefNet-matting", 1024),     # 4x faster, softer
    "general":    ("ZhengPeng7/BiRefNet", 1024),
    "portrait":   ("ZhengPeng7/BiRefNet-portrait", 1024),
    "lite":       ("ZhengPeng7/BiRefNet_lite", 1024),         # CPU-friendly
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

_LOADED = {}


def total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 16.0


def pick_size(model, ram_gb=None):
    """Largest inference resolution that fits this machine's RAM (fp16)."""
    native = MODELS[model][1]
    ram = ram_gb or total_ram_gb()
    if ram >= 30:
        cap = 2048
    elif ram >= 14:
        cap = 1536
    else:
        cap = 1024
    return min(native, cap)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model="hr-matting", device=None, half=None):
    """Load a BiRefNet checkpoint once and keep it around.

    half=None means fp16 on a GPU (mps/cuda), fp32 on CPU where half
    precision is unsupported for several ops."""
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}, pick one of: {', '.join(MODELS)}")
    device = device or pick_device()
    if half is None:
        half = device.type != "cpu"
    key = (model, str(device), half)
    if key in _LOADED:
        return _LOADED[key]

    from transformers import AutoModelForImageSegmentation

    repo, _ = MODELS[model]
    # the published checkpoints are fp16; fp16 on MPS is 3.6x faster than
    # fp32, needs half the memory and gives the same alpha (see docstring).
    dtype = torch.float16 if half else torch.float32
    net = AutoModelForImageSegmentation.from_pretrained(
        repo, trust_remote_code=True, dtype=dtype)
    net.to(device)
    net.eval()
    torch.set_float32_matmul_precision("high")
    _LOADED[key] = (net, pick_size(model), device, half)
    return _LOADED[key]


def _release_memory():
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_models():
    """Drop every loaded checkpoint and give the memory back to the OS.
    Takes ~0.3 s; the next load_model() costs ~1-4 s again."""
    _LOADED.clear()
    _release_memory()


def unload_model(model):
    """Drop one checkpoint (every device/precision variant of it), keep the rest."""
    for key in [k for k in _LOADED if k[0] == model]:
        del _LOADED[key]
    _release_memory()


# ------------------------------------------------------------------- matting

def predict_alpha(image, net, size, device, half=False, tta=False):
    """RGB PIL image -> single channel 'L' alpha at the image's own size."""
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(image).unsqueeze(0).to(device=device, dtype=next(net.parameters()).dtype)

    with torch.no_grad():
        pred = net(x)[-1].sigmoid()
        if tta:
            flipped = net(torch.flip(x, dims=[3]))[-1].sigmoid()
            pred = (pred + torch.flip(flipped, dims=[3])) / 2

    alpha = pred[0].float().cpu()
    return transforms.ToPILImage()(alpha).resize(image.size, Image.LANCZOS)


def unmix_foreground(image, alpha):
    """Estimate the true foreground colour so hair keeps no halo of the old
    background. Falls back to the raw pixels if pymatting is missing."""
    try:
        import numpy as np
        from pymatting import estimate_foreground_ml
    except ImportError:
        return image
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    a = np.asarray(alpha, dtype=np.float64) / 255.0
    fg = estimate_foreground_ml(rgb, a, return_background=False)
    return Image.fromarray((np.clip(fg, 0, 1) * 255).astype("uint8"), "RGB")


def cutout(image, net, size, device, half=False, tta=False, unmix=True):
    """RGB PIL image -> (RGBA cutout, alpha mask)."""
    alpha = predict_alpha(image, net, size, device, half=half, tta=tta)
    fg = unmix_foreground(image, alpha) if unmix else image
    out = fg.convert("RGBA")
    out.putalpha(alpha)
    return out, alpha


def flatten(rgba, background):
    """Put an RGBA cutout on a solid colour or on a background image."""
    if isinstance(background, Image.Image):
        bg = background.convert("RGB").resize(rgba.size, Image.LANCZOS)
    else:
        bg = Image.new("RGB", rgba.size, background)
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def parse_background(value):
    """'white', '#ff00aa', '255,0,0' or a path to an image."""
    if value is None:
        return None
    if os.path.isfile(value):
        return Image.open(value)
    try:
        return ImageColor.getrgb(value)
    except ValueError:
        raise SystemExit(f"cannot read background {value!r}: not a colour, not a file")


# ---------------------------------------------------------------------- cli

def is_own_output(name):
    """<name>.cutout.png / <name>.mask.png -- our own results from a previous
    run. Scanning a folder twice must not feed them back in."""
    stem = os.path.splitext(name)[0]
    return stem.endswith(".cutout") or stem.endswith(".mask")


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if os.path.splitext(name)[1].lower() not in IMAGE_EXT:
                    continue
                if is_own_output(name):
                    continue
                files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"skip (not found): {p}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Local background removal with BiRefNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("-o", "--out", help="output file (one input) or folder")
    ap.add_argument("-m", "--model", default="hr-matting", choices=list(MODELS),
                    help="checkpoint to use")
    ap.add_argument("-s", "--size", type=int,
                    help="inference resolution; default follows RAM "
                         "(2048 at 32GB+, 1536 at 16GB, 1024 below)")
    ap.add_argument("-b", "--bg", help="flatten onto this colour or image "
                                       "instead of writing transparency")
    ap.add_argument("--mask", action="store_true", help="also write <name>.mask.png")
    ap.add_argument("--jpg", action="store_true",
                    help="write jpg (implies --bg white unless --bg is given)")
    ap.add_argument("--tta", action="store_true",
                    help="average with the horizontal flip, ~2x slower")
    ap.add_argument("--no-unmix", action="store_true",
                    help="skip foreground colour estimation (faster, can halo)")
    ap.add_argument("--fp32", action="store_true",
                    help="fp32 inference (2x memory, 3-4x slower, same output)")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        raise SystemExit("nothing to do")

    background = parse_background(args.bg)
    if args.jpg and background is None:
        background = (255, 255, 255)

    out_dir = None
    if args.out and (len(files) > 1 or os.path.isdir(args.out)):
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)

    device = pick_device(args.device)
    print(f"loading {MODELS[args.model][0]} on {device} ...", flush=True)
    t0 = time.time()
    net, size, device, half = load_model(args.model, device,
                                         half=False if args.fp32 else None)
    size = args.size or size
    print(f"ready in {time.time() - t0:.1f}s, {'fp32' if not half else 'fp16'}, "
          f"running at {size}x{size} ({total_ram_gb():.0f} GB RAM)")

    for path in files:
        t0 = time.time()
        image = Image.open(path).convert("RGB")
        rgba, alpha = cutout(image, net, size, device, half=half, tta=args.tta,
                             unmix=not args.no_unmix)

        stem = os.path.splitext(os.path.basename(path))[0]
        folder = out_dir or os.path.dirname(os.path.abspath(path))
        ext = ".jpg" if args.jpg else ".png"
        if args.out and out_dir is None:
            dest = args.out
        else:
            dest = os.path.join(folder, f"{stem}.cutout{ext}")

        if background is not None:
            flatten(rgba, background).save(dest, quality=95)
        else:
            rgba.save(dest)
        if args.mask:
            alpha.save(os.path.join(folder, f"{stem}.mask.png"))
        print(f"{os.path.basename(path)} -> {dest}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
