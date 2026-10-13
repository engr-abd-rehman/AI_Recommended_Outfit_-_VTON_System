"""
app.py  -  Flask server for AI Outfit Recommendation System.

Routes:
    GET  /                          -> main UI
    GET  /api/model-image           -> serve base model photo
    GET  /catalog-image/<fn>        -> serve a DeepFashion catalog image
    POST /upload                    -> save uploaded clothing image, return filepath
    POST /api/recommend             -> CLIP-based top-4 outfit recommendations
    POST /tryon                     -> start OOTDiffusion 2-pass job, returns job_id
    GET  /tryon/progress/<job_id>   -> poll live progress (0-100 %) + result URL
"""

import os, uuid, subprocess, shutil, re, threading, json
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, abort

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
MODEL_DIR   = os.path.join(BASE_DIR, "static", "model")
CATALOG_DIR = os.path.join(BASE_DIR, "Kaggle_datasets", "train_images")
SEGM_DIR    = os.path.join(BASE_DIR, "Kaggle_datasets", "segm")
CROP_DIR    = os.path.join(BASE_DIR, "static", "crops")

# ── OOTDiffusion subprocess config ────────────────────────────────────────────
OOTD_PYTHON     = os.path.join(BASE_DIR, "ootd_env",     "Scripts", "python.exe")
OOTD_RUN_DIR    = os.path.join(BASE_DIR, "OOTDiffusion",  "run")
OOTD_RUN_SCRIPT = os.path.join(OOTD_RUN_DIR,              "run_ootd.py")
OOTD_OUT_DIR    = os.path.join(OOTD_RUN_DIR,              "images_output")
OOTD_N_SAMPLES  = 1      # 1 sample → faster, stable; output is out_hd_0 / out_dc_0
OOTD_STEPS      = 40     # diffusion steps per pass (40 gives noticeably better texture/detail) //40
OOTD_SCALE      = 2    # guidance scale (3.5 gives strong garment colour/texture adherence)//3.5

# ── Background job registry ───────────────────────────────────────────────────
_jobs      = {}           # job_id → progress dict
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kw):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR,   exist_ok=True)

# Segmentation palette label → clothing region
# val=1: upper body (~32% from top)   val=5: lower body (~63% from top)
_SEGM_LABELS = {
    "upper": [1],
    "lower": [5],
    "outer": [1],      # jackets/vests sit on upper body
    "full":  [1, 5],   # full-body garments span both
}


def _generate_crop(filename: str, garment_type: str):
    """
    Generate and cache a professional ecommerce-style clothing cutout.

    Pipeline:
      1. Apply segmentation mask to zero-out every non-clothing pixel.
      2. Feather the alpha channel with a light Gaussian blur so mask
         boundaries look smooth rather than jagged.
      3. Crop to a generously padded bounding box (15 % vertical,
         12 % horizontal) so necklines / sleeves / hems are never clipped.
      4. Scale the garment so it fills at most MAX_FILL of the fixed canvas.
      5. Centre the garment on a 320 × 420 transparent canvas so every
         recommendation card shows consistently-sized, centred clothing.

    Returns the absolute path to the cached PNG, or None on failure.
    """
    # ── Fixed canvas dimensions ───────────────────────────────────────────────
    # 360×480 gives the frontend a higher-resolution preview image.
    # The CSS .slot-img max-height (155 px) scales it down for display, but
    # browsers use the extra pixels for sharper rendering on retina screens
    # and when the card layout gives the image more room.
    CANVAS_W, CANVAS_H = 360, 480   # px (was 320×420)
    MAX_FILL = 0.82                  # garment fills <= 82 % of the smaller axis

    labels = _SEGM_LABELS.get(garment_type, [1])

    stem      = os.path.splitext(filename)[0]
    segm_file = os.path.join(SEGM_DIR,    stem + "_segm.png")
    orig_file = os.path.join(CATALOG_DIR, filename)

    cache_name = f"{stem}_{garment_type}_cut.png"
    cache_path = os.path.join(CROP_DIR, cache_name)

    if os.path.isfile(cache_path):
        return cache_path

    if not os.path.isfile(segm_file) or not os.path.isfile(orig_file):
        return None

    try:
        img      = Image.open(orig_file).convert("RGBA")
        mask_img = Image.open(segm_file)

        # Resize mask to match source if sizes differ
        if (mask_img.width, mask_img.height) != (img.width, img.height):
            mask_img = mask_img.resize((img.width, img.height), Image.NEAREST)

        mask_arr = np.array(mask_img)   # (H, W) uint8

        # ── Build binary clothing mask ────────────────────────────────────────
        binary = np.zeros(mask_arr.shape, dtype=bool)
        for lbl in labels:
            binary |= (mask_arr == lbl)

        if binary.sum() < 100:
            return None

        # ── Step 1: segmentation-based skin + background removal (ALL types) ────
        # Use the segmentation mask directly as the alpha channel.
        # Pixels NOT in the garment mask (skin, other clothes, background)
        # become fully transparent.  This works for upper (label=1) AND
        # lower (label=5): pants crops no longer show legs, shirt, or shoes.
        img_arr = np.array(img, dtype=np.uint8)   # (H, W, 4) RGBA
        img_arr[~binary, 3] = 0    # non-garment → transparent
        img_arr[binary,  3] = 255  # garment pixels → fully opaque

        masked_pil = Image.fromarray(img_arr)

        # ── Step 3: category-aware padding ───────────────────────────────────
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        y1, y2 = int(rows[0]), int(rows[-1])
        x1, x2 = int(cols[0]), int(cols[-1])

        h_box, w_box = y2 - y1, x2 - x1

        # Horizontal: 12 % on both sides (keeps sleeves / leg edges)
        pad_x = max(int(w_box * 0.12), 12)

        # Vertical — different for upper vs lower garments:
        #   Upper: 15 % top (neckline), 15 % bottom (hem)
        #   Lower: 8 % top (waistband), 3 % bottom ONLY
        #     → 3 % bottom prevents the +15 % padding from reaching the shoes,
        #       which sit just below the segmentation bottom boundary (y2).
        if garment_type == "lower":
            pad_y_top    = max(int(h_box * 0.08), 10)
            pad_y_bottom = max(int(h_box * 0.03),  5)
        else:
            pad_y_top    = max(int(h_box * 0.15), 20)
            pad_y_bottom = max(int(h_box * 0.15), 20)

        y1 = max(0,           y1 - pad_y_top)
        y2 = min(img.height,  y2 + pad_y_bottom)
        x1 = max(0,           x1 - pad_x)
        x2 = min(img.width,   x2 + pad_x)

        crop = masked_pil.crop((x1, y1, x2, y2))

        # ── Step 4: scale normalisation ───────────────────────────────────────
        cw, ch = crop.size
        scale  = min(CANVAS_W * MAX_FILL / cw,
                     CANVAS_H * MAX_FILL / ch,
                     1.0)               # never upscale beyond native resolution
        new_w  = max(1, int(cw * scale))
        new_h  = max(1, int(ch * scale))
        crop   = crop.resize((new_w, new_h), Image.LANCZOS)

        # ── Step 4b: UnsharpMask sharpening on RGB channels only ─────────────
        # Applied AFTER LANCZOS downscale so the filter recovers micro-texture
        # lost during resize.  Alpha channel is untouched to avoid edge halos.
        # This makes the garment preview look crisper in the recommendation cards.
        r_ch, g_ch, b_ch, a_ch = crop.split()
        rgb_crop  = Image.merge("RGB", (r_ch, g_ch, b_ch))
        rgb_sharp = rgb_crop.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2)
        )
        r_s, g_s, b_s = rgb_sharp.split()
        crop = Image.merge("RGBA", (r_s, g_s, b_s, a_ch))

        # ── Step 5: centre on fixed transparent canvas ────────────────────────
        canvas  = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        paste_x = (CANVAS_W - new_w) // 2
        paste_y = (CANVAS_H - new_h) // 2
        canvas.paste(crop, (paste_x, paste_y), crop)

        canvas.save(cache_path, "PNG")
        return cache_path

    except Exception:
        return None

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXT


# -- Lazy-load recommender (keeps Flask startup fast) ---------------------
_reco_fn  = None
_ready_fn = None


def _get_recommender():
    global _reco_fn, _ready_fn
    if _reco_fn is None:
        from recommender import get_recommendations, is_index_ready
        _reco_fn  = get_recommendations
        _ready_fn = is_index_ready
    return _reco_fn, _ready_fn


# -- Routes ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-image")
def model_image():
    """Serve the full-body model photo for the Virtual Try-On panel."""
    for fn in ("base_model.jpg", "base_model.png", "mannequin-base.png"):
        if os.path.isfile(os.path.join(MODEL_DIR, fn)):
            return send_from_directory(MODEL_DIR, fn)
    return "", 404


@app.route("/catalog-image/<filename>")
def catalog_image(filename):
    """Serve a DeepFashion train_images file to the frontend."""
    safe = os.path.basename(filename)          # prevent directory traversal
    if not os.path.isfile(os.path.join(CATALOG_DIR, safe)):
        abort(404)
    return send_from_directory(CATALOG_DIR, safe)


@app.route("/catalog-crop/<filename>/<garment_type>")
def catalog_crop(filename, garment_type):
    """
    Serve only the cropped clothing region (upper or lower) from a
    DeepFashion catalog image using the corresponding segm mask.
    Falls back to the full image when no mask is available.
    """
    safe = os.path.basename(filename)
    crop_path = _generate_crop(safe, garment_type)
    if crop_path and os.path.isfile(crop_path):
        return send_file(crop_path, mimetype="image/png")
    # Graceful fallback: serve the full catalog image
    if os.path.isfile(os.path.join(CATALOG_DIR, safe)):
        return send_from_directory(CATALOG_DIR, safe)
    abort(404)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        Image.open(file.stream).verify()
        file.stream.seek(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify({"error": "Please upload a valid image file"}), 400

    ext       = file.filename.rsplit(".", 1)[-1].lower()
    safe_name = f"upload_{uuid.uuid4().hex}.{ext}"
    filepath  = os.path.join(UPLOAD_DIR, safe_name)
    file.save(filepath)

    return jsonify({
        "success":  True,
        "filepath": filepath,
        "filename": safe_name,
    })


@app.route("/api/recommend", methods=["POST"])
@app.route("/recommend",     methods=["POST"])
def recommend():
    data     = request.get_json(silent=True) or {}
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No filepath provided"}), 400

    # Resolve relative paths to UPLOAD_DIR (security guard)
    if not os.path.isabs(filepath):
        filepath = os.path.join(UPLOAD_DIR, os.path.basename(filepath))

    if not os.path.isfile(filepath):
        return jsonify({"error": "Uploaded file not found on server"}), 400

    get_reco, is_ready = _get_recommender()

    if not is_ready():
        return jsonify({
            "success": False,
            "error":   "Catalog index not ready. Run:  python build_index.py"
        }), 503

    result = get_reco(filepath, top_k=4)
    return jsonify(result)


# ── CatVTON try-on helpers ────────────────────────────────────────────────────

def _find_base_model():
    """Return the absolute path of the real human model image used for CatVTON inference.

    Priority order: base_model.jpg → base_model.png
    mannequin-base.png is intentionally excluded — it is only a stylised
    placeholder for the UI preview, NOT for diffusion inference.
    """
    for fn in ("base_model.jpg", "base_model.png"):
        p = os.path.join(MODEL_DIR, fn)
        if os.path.isfile(p):
            return p
    return None


# ── OOTDiffusion: output file selector ───────────────────────────────────────

def _pick_best_ootd_output(model_type: str) -> str | None:
    """
    With OOTD_N_SAMPLES=1 there is exactly one output: out_<type>_0.png.
    Returns its path if it exists, else None.
    """
    p = os.path.join(OOTD_OUT_DIR, f"out_{model_type}_0.png")
    if os.path.isfile(p):
        print(f"[OOTD] Output found: {os.path.basename(p)}", flush=True)
        return p
    return None


# ── OOTDiffusion: progress-tracked subprocess pass ───────────────────────────

_STEP_RE = re.compile(rb'(\d+)/(\d+)')


def _stderr_reader(proc, job_id: str, p_start: int, p_end: int,
                   collected: list):
    """
    Read subprocess stderr in chunks.
    • Parses tqdm 'X/20' lines to update job progress.
    • Appends every raw chunk to `collected` so the caller can
      build a full error message even after the stream is exhausted.
    """
    buf = b""
    while True:
        chunk = proc.stderr.read(256)
        if not chunk:
            break
        collected.append(chunk)
        buf += chunk
        for part in re.split(rb'[\r\n]+', buf)[:-1]:
            m = _STEP_RE.search(part)
            if m:
                step  = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    prog = int(p_start + (step / float(total)) * (p_end - p_start))
                    _update_job(job_id, progress=prog,
                                message=f"Diffusion step {step}/{total}")
        buf = re.split(rb'[\r\n]+', buf)[-1]


def _run_ootd_pass_tracked(cloth_path: str, model_img: str,
                            model_type: str, category: int,
                            job_id: str, p_start: int, p_end: int,
                            scale: float = OOTD_SCALE) -> str:
    """Run one OOTDiffusion pass with live progress updates."""
    cmd = [
        OOTD_PYTHON, OOTD_RUN_SCRIPT,
        "--model_path", model_img,
        "--cloth_path", cloth_path,
        "--model_type", model_type,
        "--category",   str(category),
        "--scale",      str(scale),
        "--sample",     str(OOTD_N_SAMPLES),
        "--step",       str(OOTD_STEPS),
    ]
    print(f"[OOTD] {model_type} pass — model: {os.path.basename(model_img)}"
          f" | cloth: {os.path.basename(cloth_path)}", flush=True)

    collected = []   # stderr chunks stored by reader thread
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=OOTD_RUN_DIR,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
    )
    reader = threading.Thread(target=_stderr_reader,
                              args=(proc, job_id, p_start, p_end, collected),
                              daemon=True)
    reader.start()
    proc.wait()
    reader.join(timeout=5)

    if proc.returncode != 0:
        # Use what the reader thread already collected — proc.stderr is drained
        err = b"".join(collected).decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"OOTDiffusion pass '{model_type}' failed:\n{err}")

    best = _pick_best_ootd_output(model_type)
    if best is None:
        raise RuntimeError("OOTDiffusion produced no output files.")
    return best


# ── OOTDiffusion: garment normalizer ─────────────────────────────────────────

def _catalog_garment_hires(filename: str, garment_type: str,
                            dest_dir: str, tag: str):
    """
    Extract the garment from a catalog image at OOTDiffusion-native resolution
    (768 × 1024) instead of the 360 × 480 display canvas used by _generate_crop.

    The 360→768 upscaling that happened previously caused:
      • Severe colour washing / saturation loss  (maroon → pink)
      • Texture blurring and detail loss
      • Interpolation artefacts that confused the diffusion model

    Here we apply the segmentation mask at the ORIGINAL image resolution,
    crop to the garment bounding box, and downscale (if needed) or upscale
    with Lanczos to fit an 85 %-fill target inside 768 × 1024, saving at
    JPEG quality=97 so the diffusion model receives maximum colour fidelity.

    Returns the absolute path of the ready JPEG, or None on failure.
    """
    labels = _SEGM_LABELS.get(garment_type, [1])
    stem   = os.path.splitext(filename)[0]
    out    = os.path.join(dest_dir, f"_ootd_{tag}_{stem}_hires.jpg")

    if os.path.isfile(out):          # already generated
        return out

    segm_file = os.path.join(SEGM_DIR,    stem + "_segm.png")
    orig_file = os.path.join(CATALOG_DIR, filename)

    if not os.path.isfile(segm_file) or not os.path.isfile(orig_file):
        return None

    try:
        img      = Image.open(orig_file).convert("RGBA")
        mask_img = Image.open(segm_file)

        if (mask_img.width, mask_img.height) != (img.width, img.height):
            mask_img = mask_img.resize((img.width, img.height), Image.NEAREST)

        mask_arr = np.array(mask_img)           # (H, W) uint8 or (H, W, C)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]         # use first channel if RGB

        binary = np.zeros(mask_arr.shape, dtype=bool)
        for lbl in labels:
            binary |= (mask_arr == lbl)

        if binary.sum() < 200:
            print(f"[OOTD] {tag}: hi-res — too few garment pixels, skipping", flush=True)
            return None

        # Apply mask — garment pixels opaque, everything else transparent
        img_arr = np.array(img, dtype=np.uint8)
        img_arr[~binary, 3] = 0
        img_arr[binary,  3] = 255

        # Bounding box with conservative padding
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        y1, y2 = int(rows[0]), int(rows[-1])
        x1, x2 = int(cols[0]), int(cols[-1])
        H, W   = img_arr.shape[:2]

        if garment_type == "lower":
            pad_y_top    = max(int((y2 - y1) * 0.08), 8)
            pad_y_bottom = max(int((y2 - y1) * 0.03), 4)
        else:
            pad_y_top    = max(int((y2 - y1) * 0.12), 12)
            pad_y_bottom = max(int((y2 - y1) * 0.12), 12)

        pad_x = max(int((x2 - x1) * 0.10), 10)
        y1 = max(0, y1 - pad_y_top);   y2 = min(H, y2 + pad_y_bottom)
        x1 = max(0, x1 - pad_x);       x2 = min(W, x2 + pad_x)

        cropped = Image.fromarray(img_arr).crop((x1, y1, x2, y2))
        cw, ch  = cropped.size

        # Scale to 85 % fill within 768 × 1024, preserving aspect ratio
        TARGET_W, TARGET_H = 768, 1024
        fill = 0.85
        scale = min(TARGET_W * fill / cw, TARGET_H * fill / ch)
        new_w  = max(1, int(cw * scale))
        new_h  = max(1, int(ch * scale))
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # Composite onto white 768 × 1024 canvas (centre)
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
        px = (TARGET_W - new_w) // 2
        py = (TARGET_H - new_h) // 2
        canvas.paste(resized, (px, py), mask=resized.split()[3])

        canvas.save(out, "JPEG", quality=97)
        print(f"[OOTD] {tag}: hi-res garment → {os.path.basename(out)} "
              f"({new_w}×{new_h} on {TARGET_W}×{TARGET_H})", flush=True)
        return out

    except Exception as exc:
        print(f"[OOTD] {tag}: hi-res extraction error: {exc}", flush=True)
        return None


def _normalize_for_ootd(src_path: str, dest_dir: str, tag: str) -> str:
    """
    Convert any uploaded/catalog image to a clean RGB JPEG that PIL (inside
    OOTDiffusion's run_ootd.py) can open without UnidentifiedImageError.

    Handles: RGBA PNG, WebP, HEIC-renamed, partial uploads, etc.
    Returns the absolute path of the normalised JPEG.
    """
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out  = os.path.join(dest_dir, f"_ootd_{tag}_{stem}.jpg")

    img = Image.open(src_path)          # raises if file is corrupt / 0-byte
    if img.mode in ("RGBA", "LA", "P"):
        # Flatten transparency onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    else:
        img = img.convert("RGB")

    img.save(out, "JPEG", quality=95)
    print(f"[OOTD] Normalised {tag}: {os.path.basename(src_path)} → {os.path.basename(out)}", flush=True)
    return out


def _prepare_garment_for_ootd(garment_path: str, garment_type: str,
                               dest_dir: str, tag: str) -> str:
    """
    Prepare a garment image for OOTDiffusion at its native 768×1024 resolution.

    • Catalog images  → _catalog_garment_hires() extracts the garment at full
      original resolution and places it on a 768×1024 white canvas.
      This eliminates the 2× upscaling artefacts (washed-out colours, blurry
      textures) that occurred when the 360×480 display crop was fed to the
      diffusion model.  Falls back to the display crop → white BG if hi-res
      extraction fails, then to the raw catalog image.

    • User-uploaded images  → already clean garment shots; just normalise to
      RGB JPEG via _normalize_for_ootd.

    Returns the absolute path of the ready-to-use JPEG.
    """
    is_catalog = os.path.normpath(garment_path).startswith(
        os.path.normpath(CATALOG_DIR)
    )

    if is_catalog:
        filename = os.path.basename(garment_path)

        # ── Primary: hi-res garment at 768×1024 (best colour/texture fidelity) ─
        hires = _catalog_garment_hires(filename, garment_type, dest_dir, tag)
        if hires:
            return hires

        # ── Fallback 1: display crop (360×480) pasted on white ────────────────
        print(f"[OOTD] {tag}: hi-res failed, falling back to display crop", flush=True)
        crop_path = _generate_crop(filename, garment_type)
        if crop_path and os.path.isfile(crop_path):
            stem = os.path.splitext(filename)[0]
            out  = os.path.join(dest_dir, f"_ootd_{tag}_{stem}.jpg")
            crop = Image.open(crop_path).convert("RGBA")
            bg   = Image.new("RGB", crop.size, (255, 255, 255))
            bg.paste(crop, mask=crop.split()[3])
            bg.save(out, "JPEG", quality=95)
            print(f"[OOTD] {tag}: display crop → {os.path.basename(out)}", flush=True)
            return out

        # ── Fallback 2: raw catalog image (no segmentation) ───────────────────
        print(f"[OOTD] {tag}: no segmentation mask, using raw catalog image", flush=True)

    # User upload or catalog fallback: standard normalisation
    return _normalize_for_ootd(garment_path, dest_dir, tag)


# ── OOTDiffusion: upper-body compositor ──────────────────────────────────────

def _composite_upper_onto_lower(upper_path: str, lower_path: str,
                                 split_ratio: float = 0.46) -> str:
    """
    After the two OOTDiffusion passes, the DC model (lower body) can subtly
    change the shirt appearance — colours shift, sleeves lengthen, a tan band
    appears at the waist.

    This function restores the EXACT upper body from Pass 1 by blending:
      • 0 … (split_y − feather) : taken entirely from upper_path  (shirt exact)
      • blend zone              : linear cross-fade over 5 % of height
      • (split_y + feather) … H : taken entirely from lower_path  (pants/shorts)

    split_ratio = 0.46 puts the blend boundary just above the trouser waistband
    (~mid-torso), which hides the blend and keeps both garments fully intact.

    Returns the path to the composited PNG (written next to lower_path).
    """
    upper = Image.open(upper_path).convert("RGB")
    lower = Image.open(lower_path).convert("RGB")

    if upper.size != lower.size:
        lower = lower.resize(upper.size, Image.LANCZOS)

    _, H = upper.size
    split_y  = int(H * split_ratio)
    feather  = max(int(H * 0.04), 10)   # 4 % of height as blend zone

    u = np.array(upper, dtype=np.float32)
    l = np.array(lower, dtype=np.float32)
    out = l.copy()

    # Pure upper-body region — shirt taken verbatim from Pass 1
    top = max(0, split_y - feather)
    out[:top] = u[:top]

    # Feathered blend zone
    for i in range(feather * 2):
        y = split_y - feather + i
        if 0 <= y < H:
            alpha = i / (feather * 2)          # 0 = full upper, 1 = full lower
            out[y] = u[y] * (1.0 - alpha) + l[y] * alpha

    # Lower region already holds lower_path content from the copy()

    comp_path = lower_path.replace(".png", "_comp.png")
    Image.fromarray(out.astype(np.uint8)).save(comp_path, "PNG")
    print(f"[OOTD] Upper-body composite saved → {os.path.basename(comp_path)}", flush=True)
    return comp_path


# ── OOTDiffusion: background job ─────────────────────────────────────────────

def _ootd_background_job(job_id: str, shirt_path: str, pants_path: str,
                          output_path: str):
    """
    Sequential 2-pass OOTDiffusion full-outfit pipeline:

    Pass 1 — Upper body (HD model, category 0)
        model_path : base_model.jpg
        cloth_path : uploaded shirt (hi-res, 768×1024)
        output     : intermediate_upper.png  (saved in UPLOAD_DIR)

    Pass 2 — Lower body (DC model, category 1)
        model_path : intermediate_upper.png  (Pass-1 result)
        cloth_path : recommended pants/shorts (hi-res, 768×1024)
        output     : raw_lower.png

    Post-process — Pass 2 output used directly (no composite)
        The DC model at scale=3.5 preserves the shirt from Pass 1 while also
        naturally removing base-model white-sleeve/undershirt artefacts.
        output     : final_outfit.png  (copied to output_path)
    """
    # Fixed debug/output filenames for easy inspection
    intermediate_path = os.path.join(UPLOAD_DIR, "intermediate_upper.png")
    final_named_path  = os.path.join(UPLOAD_DIR, "final_outfit.png")

    try:
        if not os.path.isfile(OOTD_PYTHON):
            raise RuntimeError(f"ootd_env Python not found: {OOTD_PYTHON}")

        mannequin = _find_base_model()
        if mannequin is None:
            raise RuntimeError("Base model image not found in static/model/.")

        # ── Pre-flight: prepare garments at 768×1024 (full OOTDiffusion res) ───
        # _prepare_garment_for_ootd now generates hi-res crops via
        # _catalog_garment_hires, preventing the 2× upscale colour/texture loss
        # that caused washed-out colours (maroon → pink) and blurry textures.
        _update_job(job_id, progress=2, message="Preparing garment images…")
        shirt_jpeg = _prepare_garment_for_ootd(shirt_path, "upper", UPLOAD_DIR, "shirt")
        pants_jpeg = _prepare_garment_for_ootd(pants_path, "lower", UPLOAD_DIR, "pants")

        # ── Pass 1: shirt on base model (HD, 5 → 50 %) ───────────────────────
        _update_job(job_id, progress=5,
                    message="Pass 1 — applying shirt (HD model)…")
        print(f"[OOTD] Pass 1: shirt={os.path.basename(shirt_jpeg)}", flush=True)

        upper_raw = _run_ootd_pass_tracked(
            cloth_path=shirt_jpeg,
            model_img=mannequin,
            model_type="hd",
            category=0,
            job_id=job_id,
            p_start=5, p_end=50,
        )

        # Save intermediate so Pass 2 can use it as model_path
        shutil.copy2(upper_raw, intermediate_path)
        print(f"[OOTD] Intermediate saved → intermediate_upper.png", flush=True)
        _update_job(job_id, progress=52,
                    message="Pass 1 complete — applying pants (DC model)…")

        # ── Pass 2: pants on shirt result (DC, 52 → 95 %) ────────────────────
        print(f"[OOTD] Pass 2: pants={os.path.basename(pants_jpeg)}", flush=True)

        final_raw = _run_ootd_pass_tracked(
            cloth_path=pants_jpeg,
            model_img=intermediate_path,   # ← shirt result used as model
            model_type="dc",
            category=1,
            job_id=job_id,
            p_start=52, p_end=95,
        )

        # ── Save final result (Pass 2 raw — no composite) ────────────────────
        # Using Pass 2 directly avoids locking in the white-sleeve / undershirt
        # artefact that comes from the base model's own clothing being visible
        # through the first-pass HD result. The DC model at scale 3.5 already
        # preserves the shirt colour/texture while cleaning the upper region.
        _update_job(job_id, progress=98, message="Saving final outfit…")
        shutil.copy2(final_raw, final_named_path)
        shutil.copy2(final_raw, output_path)
        print(f"[OOTD] Final saved → final_outfit.png", flush=True)

        result_url = "/static/uploads/" + os.path.basename(output_path)
        _update_job(job_id, status="done", progress=100,
                    message="Complete!", result_url=result_url)

    except Exception as exc:
        print(f"[OOTD] Background job {job_id} FAILED: {exc}", flush=True)
        _update_job(job_id, status="error", error=str(exc))


# ── Routes: try-on ────────────────────────────────────────────────────────────

@app.route("/tryon", methods=["POST"])
def tryon():
    """
    Start the 2-pass sequential OOTDiffusion full-outfit job.

    Request JSON:
        uploaded_filepath  – absolute path of the uploaded shirt image
        catalog_filename   – bare filename of the recommended pants/shorts
        uploaded_type      – "upper" (shirt) or "lower" (pants) — determines
                             which is the user garment vs. the catalog garment

    Response JSON: { "job_id": "<hex>" }
    Poll /tryon/progress/<job_id> for live updates.
    """
    data = request.get_json(silent=True) or {}

    uploaded_filepath = data.get("uploaded_filepath", "").strip()
    catalog_filename  = data.get("catalog_filename",  "").strip()
    uploaded_type     = data.get("uploaded_type",     "upper").strip()

    if not uploaded_filepath:
        return jsonify({"error": "uploaded_filepath is required"}), 400
    if not catalog_filename:
        return jsonify({"error": "catalog_filename is required"}), 400

    # Resolve uploaded path
    if not os.path.isabs(uploaded_filepath):
        uploaded_filepath = os.path.join(UPLOAD_DIR, os.path.basename(uploaded_filepath))
    if not os.path.isfile(uploaded_filepath):
        return jsonify({"error": "Uploaded file not found on server"}), 400

    # Resolve catalog pants path
    catalog_path = os.path.join(CATALOG_DIR, os.path.basename(catalog_filename))
    if not os.path.isfile(catalog_path):
        return jsonify({"error": f"Catalog item not found: {catalog_filename}"}), 400

    # Assign shirt / pants correctly based on which garment the user uploaded
    if uploaded_type == "lower":
        # User uploaded pants → catalog item is the shirt
        shirt_path = catalog_path
        pants_path = uploaded_filepath
    else:
        # User uploaded shirt (default) → catalog item is the pants
        shirt_path = uploaded_filepath
        pants_path = catalog_path

    result_name = f"tryon_{uuid.uuid4().hex}.png"
    output_path = os.path.join(UPLOAD_DIR, result_name)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        if any(job.get("status") == "running" for job in _jobs.values()):
            return jsonify({"error": "Another try-on is running. Wait for it to finish."}), 409
        _jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting…"}

    t = threading.Thread(
        target=_ootd_background_job,
        args=(job_id, shirt_path, pants_path, output_path),
        daemon=True,
    )
    try:
        t.start()
    except RuntimeError as exc:
        _update_job(job_id, status="error", error=str(exc))
        return jsonify({"error": "Could not start the try-on worker"}), 500

    print(f"[TRYON] Job {job_id} — shirt: {os.path.basename(shirt_path)}"
          f" | pants: {os.path.basename(pants_path)}", flush=True)
    return jsonify({"job_id": job_id})


@app.route("/tryon/progress/<job_id>")
def tryon_progress(job_id):
    """Return current progress of a try-on job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job ID"}), 404
    return jsonify(job)


# -- Run ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
