"""
Click-to-select on top of rmbg.py, with SAM 2 (Meta, Apache-2.0).

BiRefNet decides by itself what the subject is. When a photo has several
things in it, SAM 2 lets the user point at the one to keep: positive clicks
(label 1) inside the object, negative ones (label 0) on parts to drop. SAM's
own edges are coarse (no hair, no fur), so the final matte still comes from
BiRefNet: the picked object is cropped out with a margin, BiRefNet runs on
the crop (where the object is the obvious subject, at a higher effective
resolution), and its alpha is kept only inside the SAM region. If BiRefNet
does not see the object at all (it was background for it) the SAM mask is
used as is.

SAM 2 small is 39M parameters and runs on the CPU: ~1 s to embed an image
once, ~30 ms per click after that. Embeddings are cached per image key so
clicking around stays instant.
"""

import threading

import numpy as np
import torch
from PIL import Image

import rmbg

MODEL_ID = "facebook/sam2.1-hiera-small"
NAME = "sam2"                      # how it shows up in the server's model chips

_SAM = {}                          # "model" -> (model, processor)
_EMB = {}                          # image key -> (embeddings, (w, h))
_LOCK = threading.Lock()
PREVIEW_MAX = 1600                 # long side of the mask sent to the browser


def loaded():
    return "model" in _SAM


def load():
    with _LOCK:
        if "model" not in _SAM:
            from transformers import Sam2Model, Sam2Processor

            model = Sam2Model.from_pretrained(MODEL_ID).to("cpu").eval()
            _SAM["model"] = (model, Sam2Processor.from_pretrained(MODEL_ID))
        return _SAM["model"]


def unload():
    with _LOCK:
        _SAM.clear()
        _EMB.clear()
    import gc
    gc.collect()


def segment(image, points, key=None):
    """RGB PIL image + [[x, y, label], ...] in image pixels -> float32 HxW
    mask logits (>0 is object), at the image's own size."""
    model, proc = load()
    with _LOCK:
        emb = None
        if key is not None and key in _EMB and _EMB[key][1] == image.size:
            emb = _EMB[key][0]
        with torch.no_grad():
            if emb is None:
                px = proc(images=image, return_tensors="pt")["pixel_values"]
                emb = model.get_image_embeddings(px)
                if key is not None:
                    _EMB[key] = (emb, image.size)
                    for old in list(_EMB)[:-3]:      # keep the last three images
                        del _EMB[old]
            inp = proc(images=image,
                       input_points=[[[[float(x), float(y)] for x, y, _ in points]]],
                       input_labels=[[[int(l) for _, _, l in points]]],
                       return_tensors="pt")
            out = model(image_embeddings=emb, input_points=inp["input_points"],
                        input_labels=inp["input_labels"], multimask_output=False)
            masks = proc.post_process_masks(out.pred_masks, inp["original_sizes"],
                                            binarize=False)[0]
    return masks[0, 0].float().numpy()


def preview_png(logits, colour=(60, 140, 255)):
    """The mask as an RGBA image for the browser overlay: colour with the
    soft mask as alpha, downscaled so a click never waits on a 12 MP PNG."""
    h, w = logits.shape
    soft = 1 / (1 + np.exp(-logits))
    a = Image.fromarray((soft * 255).astype("uint8"))
    scale = min(1.0, PREVIEW_MAX / max(w, h))
    if scale < 1:
        a = a.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    out = Image.new("RGBA", a.size, colour + (0,))
    out.putalpha(a)
    return out


def cutout_picked(image, points, net, size, device, half=False, tta=False, key=None):
    """RGB PIL image -> (RGBA cutout, alpha) of the object under the clicks."""
    from scipy.ndimage import distance_transform_edt

    logits = segment(image, points, key)
    binm = logits > 0
    if not binm.any():
        raise ValueError("nothing selected at these points")
    W, H = image.size
    ys, xs = np.nonzero(binm)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(0.15 * max(x1 - x0, y1 - y0)) + 16
    box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))

    crop = image.crop(box)
    a = np.asarray(rmbg.predict_alpha(crop, net, size, device, half=half, tta=tta),
                   dtype=np.float32) / 255
    m = binm[box[1]:box[3], box[0]:box[2]]
    r = max(3, int(0.015 * max(W, H)))
    region = distance_transform_edt(~m) <= r          # SAM mask grown by r px
    core = distance_transform_edt(m) > r              # and shrunk by r px
    inside = core if core.any() else m
    if a[inside].mean() >= 0.5:
        # BiRefNet sees the object: its fine edges, limited to the SAM region,
        # with the SAM core filled in case BiRefNet punched holes in it
        out = np.maximum(a * region, core.astype(np.float32))
    else:
        # BiRefNet does not: SAM's own (soft) mask, coarse edges but correct
        lg = logits[box[1]:box[3], box[0]:box[2]]
        out = 1 / (1 + np.exp(-lg))

    full = np.zeros((H, W), dtype=np.float32)
    full[box[1]:box[3], box[0]:box[2]] = out
    alpha = Image.fromarray((full * 255).astype("uint8"))
    fg = rmbg.unmix_foreground(image, alpha)
    rgba = fg.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba, alpha
