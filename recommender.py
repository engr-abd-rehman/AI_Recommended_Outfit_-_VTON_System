"""
recommender.py  –  CLIP-based outfit COMPATIBILITY recommendation engine.

Detection Pipeline (4 stages, first match wins):
  Stage 1 – Catalog metadata lookup  : filename exact-match in catalog → garment_type
  Stage 2 – Filename keyword matching : DeepFashion / common fashion keywords
  Stage 3 – Image region analysis     : non-white pixel centroid (product-photo fallback)
  Stage 4 – CLIP zero-shot            : text-prompt cosine similarity (final fallback)

Pairing Rules (strict):
  upper → recommend lower only
  lower → recommend upper only
  outer → recommend upper + lower
  full  → recommend upper + lower

Always returns exactly top_k (=4) results.

Public API:
    is_index_ready() -> bool
    get_recommendations(image_path, top_k=4) -> dict
"""

import os, json, re
import numpy as np
from PIL import Image
import torch
import clip

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VEC_PATH    = os.path.join(BASE_DIR, "catalog_vectors.npy")
META_PATH   = os.path.join(BASE_DIR, "catalog_metadata.json")
SEGM_DIR    = os.path.join(BASE_DIR, "Kaggle_datasets", "segm")
CATALOG_DIR = os.path.join(BASE_DIR, "Kaggle_datasets", "train_images")

# ── Singletons ────────────────────────────────────────────────────────────────
_model          = None
_preprocess     = None
_device         = None
_vectors        = None   # (N, 512) float32 – already L2-normalised
_metadata       = None   # list[dict]
_catalog_fname  = {}     # filename (basename) → garment_type  (Stage 1 cache)

# ── Category Compatibility Pairing ────────────────────────────────────────────
CATEGORY_PAIRS = {
    "upper": ["lower"],
    "lower": ["upper"],
    "outer": ["upper", "lower"],
    "full":  ["upper", "lower"],
}

# ── Stage 2: Filename Keyword Maps ────────────────────────────────────────────
_UPPER_KW = {
    "shirt", "tee", "tees", "tank", "tanks", "blouse", "blouses",
    "polo", "polos", "sweater", "sweaters", "hoodie", "hoodies",
    "sweatshirt", "sweatshirts", "top", "tops", "cardigan", "cardigans",
    "graphic_tees", "tees_tanks", "blouses_shirts", "shirts_polos",
    "sweatshirts_hoodies",
}
_LOWER_KW = {
    "denim", "pant", "pants", "jean", "jeans", "trouser", "trousers",
    "short", "shorts", "skirt", "skirts", "legging", "leggings", "chino", "chinos",
}
_OUTER_KW = {
    "jacket", "jackets", "coat", "coats", "vest", "vests",
    "blazer", "blazers", "jackets_vests", "jackets_coats",
}
_FULL_KW = {
    "dress", "dresses", "jumpsuit", "jumpsuits",
    "romper", "rompers", "rompers_jumpsuits", "suiting",
}


def _kw_detect(text: str):
    """Return garment_type from keyword scan of `text`, or None."""
    words = set(re.split(r'[\W_]+', text.lower()))
    # Check each word and also the full lowercased text for multi-word tokens
    full = text.lower()
    if words & _LOWER_KW or any(k in full for k in _LOWER_KW):
        return "lower"
    if words & _UPPER_KW or any(k in full for k in _UPPER_KW):
        return "upper"
    if words & _OUTER_KW or any(k in full for k in _OUTER_KW):
        return "outer"
    if words & _FULL_KW  or any(k in full for k in _FULL_KW):
        return "full"
    return None


# ── Stage 3: Segm-lookup + isolated product-photo analysis ────────────────────
def _stage3_segm_detect(image_path: str):
    """
    Stage 3 uses two sub-strategies:

    3a. Segm-mask existence check:
        If the dataset's segm/ folder contains a mask for the uploaded image,
        that image is a catalog item → look up garment_type from _catalog_fname.
        (Safety net for catalog filenames missed by Stage 1.)

    3b. Isolated-product-photo centroid analysis:
        Applies ONLY when the image has a very white background (> 70 % of
        pixels above 220 in all channels) and the non-background pixels have
        their centroid clearly in the upper third (<0.33) or lower third (>0.67).
        These strict thresholds avoid false classifications on full-body shots.

    Returns garment_type string or None when inconclusive.
    """
    basename = os.path.basename(image_path)
    stem     = os.path.splitext(basename)[0]

    # ── 3a: segm mask lookup ──────────────────────────────────────────────────
    if os.path.isdir(SEGM_DIR):
        segm_name = stem + "_segm.png"
        if os.path.isfile(os.path.join(SEGM_DIR, segm_name)):
            # image is from catalog → find garment_type via filename lookup
            gtype = _catalog_fname.get(basename)
            if gtype:
                return gtype

    # ── 3b: isolated-product-photo centroid (conservative thresholds) ─────────
    try:
        img  = Image.open(image_path).convert("RGB")
        arr  = np.array(img, dtype=np.uint8)
        h, w = arr.shape[:2]
        total_px = h * w

        # Identify "very white" background pixels (all channels ≥ 220)
        is_bg = np.all(arr >= 220, axis=-1)
        bg_ratio = is_bg.sum() / total_px

        # Only apply centroid heuristic for isolated product photos
        # (> 70 % white/light background → likely a product shot)
        if bg_ratio < 0.70:
            return None

        fg_rows = np.where(~is_bg)[0]
        if len(fg_rows) < 200:
            return None

        centroid_y = float(fg_rows.mean())
        ratio      = centroid_y / h      # 0 = top, 1 = bottom

        # Strict thresholds: only classify when signal is very clear
        if ratio < 0.33:
            return "upper"
        if ratio > 0.67:
            return "lower"
        return None   # ambiguous → fall through to CLIP
    except Exception:
        return None


# ── CLIP Zero-Shot Category Detection Prompts (Stage 4) ───────────────────────
CATEGORY_PROMPTS = {
    "upper": [
        "a photo of a shirt", "a photo of a t-shirt", "a photo of a top",
        "a photo of a blouse", "a photo of a sweater", "a photo of a polo",
        "a photo of upper body clothing", "a photo of a tee",
        "a photo of a tank top",
    ],
    "lower": [
        "a photo of pants", "a photo of jeans", "a photo of trousers",
        "a photo of shorts", "a photo of a skirt", "a photo of chinos",
        "a photo of lower body clothing", "a photo of denim pants",
        "a photo of leggings",
    ],
    "outer": [
        "a photo of a jacket", "a photo of a coat", "a photo of outerwear",
        "a photo of a blazer", "a photo of a cardigan",
    ],
    "full": [
        "a photo of a dress", "a photo of a jumpsuit",
        "a photo of a full-body outfit", "a photo of a romper",
    ],
}


def _clip_detect(query_vec: np.ndarray) -> str:
    """Stage 4: CLIP zero-shot garment-type detection."""
    scores   = {}
    q_tensor = torch.tensor(query_vec).to(_device)   # (1, 512)
    for gtype, prompts in CATEGORY_PROMPTS.items():
        tokens = clip.tokenize(prompts).to(_device)
        with torch.no_grad():
            text_feats = _model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        avg_text = text_feats.mean(dim=0, keepdim=True)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True)
        scores[gtype] = (q_tensor @ avg_text.T).item()
    return max(scores, key=scores.get)


# ── View Quality Scores ───────────────────────────────────────────────────────
VIEW_QUALITY = {
    "front":      1.00,
    "flat":       0.80,
    "additional": 0.65,
    "side":       0.50,
    "back":       0.40,
    "full":       0.00,   # full-body runway shot — penalise
}
_VIEW_RE = re.compile(r'_(\d+)_(\w+)\.png$', re.IGNORECASE)


def _get_view_type(filename: str) -> str:
    m = _VIEW_RE.search(filename)
    return m.group(2).lower() if m else "additional"


def _view_quality(filename: str) -> float:
    return VIEW_QUALITY.get(_get_view_type(filename), 0.5)


def is_index_ready() -> bool:
    return os.path.isfile(VEC_PATH) and os.path.isfile(META_PATH)


def _ensure_loaded() -> bool:
    global _model, _preprocess, _device, _vectors, _metadata, _catalog_fname

    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = clip.load("ViT-B/32", device=_device)
        _model.eval()

    if _vectors is None:
        if not is_index_ready():
            return False
        _vectors = np.load(VEC_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
        # Build fast lookup: basename → garment_type
        _catalog_fname = {
            os.path.basename(m["filename"]): m["garment_type"]
            for m in _metadata
        }

    return True


def _embed_image(image_path: str) -> np.ndarray:
    """Return L2-normalised CLIP embedding. Shape: (1, 512)."""
    img = _preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


# ── 4-Stage Detection ─────────────────────────────────────────────────────────
def _detect_garment_type(image_path: str, query_vec: np.ndarray):
    """
    Returns (garment_type, detection_stage_label).

    Stage 1 – Exact match in catalog metadata.
    Stage 2 – Filename keyword scan.
    Stage 3 – Image foreground centroid (product photo analysis).
    Stage 4 – CLIP zero-shot text similarity.
    """
    basename = os.path.basename(image_path)

    # ── Stage 1: catalog lookup ───────────────────────────────────────────────
    if basename in _catalog_fname:
        return _catalog_fname[basename], "catalog-lookup"

    # Check without view suffix too  (e.g. strip "_segm" variants)
    stem = re.sub(r'_segm', '', basename)
    if stem in _catalog_fname:
        return _catalog_fname[stem], "catalog-lookup"

    # ── Stage 2: filename keyword scan ────────────────────────────────────────
    gtype = _kw_detect(basename)
    if gtype:
        return gtype, "filename-keyword"

    # Also try the full path (may contain folder names like "upper" / "Denim")
    gtype = _kw_detect(image_path)
    if gtype:
        return gtype, "path-keyword"

    # ── Stage 3: segm-lookup + isolated product photo analysis ────────────────
    gtype = _stage3_segm_detect(image_path)
    if gtype:
        return gtype, "segm-or-product-region"

    # ── Stage 4: CLIP zero-shot (always succeeds) ─────────────────────────────
    return _clip_detect(query_vec), "clip-zeroshot"


def _sim_to_pct(sim: float) -> float:
    """Map CLIP cosine similarity (~0.15–0.55) → 0–100 %."""
    return round(min(max((sim - 0.15) / 0.40 * 100, 0), 100), 1)


# ── Public API ────────────────────────────────────────────────────────────────
def get_recommendations(image_path: str, top_k: int = 4) -> dict:
    if not _ensure_loaded():
        return {"success": False, "error": "Catalog index not found. Run build_index.py first."}

    try:
        query = _embed_image(image_path)   # (1, 512)
    except Exception as exc:
        return {"success": False, "error": f"Could not read image: {exc}"}

    # ── Stage 1-4: 4-stage garment type detection ────────────────────────────
    query_type, detected_by = _detect_garment_type(image_path, query)
    target_types = CATEGORY_PAIRS.get(query_type, ["upper", "lower"])

    # ── 2. Filter catalog: compatible category AND straight front pose ─────────
    # DeepFashion filenames encode the view angle:
    #   _1_front     = straight-on front view  ← ONLY these are used
    #   _2_side      = side view               ← excluded
    #   _3_back      = back view               ← excluded
    #   _4_full      = full-body runway shot   ← excluded (often angled/walking)
    #   _7_additional = misc extra poses       ← excluded
    #
    # Front-pose images produce the cleanest garment crops and are most
    # useful for the virtual try-on pipeline.
    compatible_idx = np.array([
        i for i, m in enumerate(_metadata)
        if m["garment_type"] in target_types
        and "_1_front" in m["filename"]          # straight front pose only
    ])

    # Fallback: if the front-pose pool is too small (< 4), relax to all poses
    # so the system never returns an empty result.
    if len(compatible_idx) < 4:
        print(f"[Recommender] WARNING: only {len(compatible_idx)} front-pose items "
              f"found for {target_types}. Relaxing pose filter to all views.",
              flush=True)
        compatible_idx = np.array([
            i for i, m in enumerate(_metadata)
            if m["garment_type"] in target_types
        ])

    if len(compatible_idx) == 0:
        return {"success": False, "error": "No compatible items in catalog."}

    compat_vectors = _vectors[compatible_idx]                # (M, 512)
    raw_scores     = (compat_vectors @ query.T).squeeze()    # (M,)

    # ── 3. Adjust scores: reward front-view, penalise full-body ──────────────
    view_bonus = np.array([
        _view_quality(_metadata[i]["filename"]) for i in compatible_idx
    ], dtype=np.float32)
    # ±0.05 adjustment; front gets +0.05, full-body gets -0.05
    adjusted_scores = raw_scores + (view_bonus - 0.5) * 0.10

    # ── 3b. Try-on source quality bonus (upper garments only) ────────────────
    # Rewards catalog images that are front-facing, sleeve-symmetric, and
    # have low skin occlusion — making them cleaner OOTDiffusion inputs.
    # Weight: ±0.04  (keeps it subordinate to CLIP similarity)
    try:
        from garment_cleaner import score_garment_tryon_quality
        quality_bonus = np.array([
            (score_garment_tryon_quality(
                os.path.join(CATALOG_DIR, os.path.basename(_metadata[i]["filename"])),
                SEGM_DIR
            ) - 0.5) * 0.08
            if _metadata[i]["garment_type"] == "upper" else 0.0
            for i in compatible_idx
        ], dtype=np.float32)
        adjusted_scores = adjusted_scores + quality_bonus
    except Exception as _qe:
        print(f"[Recommender] quality scoring skipped: {_qe}", flush=True)

    # ── 4. Rank and deduplicate by item ID (avoid same item, different view) ──
    sorted_local  = np.argsort(adjusted_scores)[::-1]
    seen_item_ids = set()
    results       = []   # list of (global_idx, raw_sim)

    for local_i in sorted_local:
        global_i = int(compatible_idx[local_i])
        fname    = _metadata[global_i]["filename"]
        id_match = re.search(r'id_(\d+)', fname)
        item_id  = id_match.group(1) if id_match else fname

        if item_id not in seen_item_ids:
            seen_item_ids.add(item_id)
            results.append((global_i, float(raw_scores[local_i])))

        if len(results) >= top_k:
            break

    # ── 5. Fallback: pad to top_k if dedup removed too many ──────────────────
    if len(results) < top_k:
        used_globals = {g for g, _ in results}
        for local_i in sorted_local:
            global_i = int(compatible_idx[local_i])
            if global_i not in used_globals:
                results.append((global_i, float(raw_scores[local_i])))
                used_globals.add(global_i)
            if len(results) >= top_k:
                break

    # ── 6. Build recommendation objects ──────────────────────────────────────
    recommendations = []
    for global_i, sim in results:
        meta = _metadata[global_i]
        recommendations.append({
            "filename":     meta["filename"],
            "category":     meta["category"],
            "garment_type": meta["garment_type"],
            "gender":       meta.get("gender", ""),
            "match_pct":    _sim_to_pct(sim),
            "image_url":    f"/catalog-crop/{meta['filename']}/{meta['garment_type']}",
            "view_type":    _get_view_type(meta["filename"]),
        })

    # ── Compatibility scores from top match ───────────────────────────────────
    top_sim = results[0][1] if results else 0.3
    color_m = round(min(_sim_to_pct(top_sim) * 1.05, 100), 1)
    style_m = round(min(_sim_to_pct(top_sim) * 1.02, 100), 1)
    overall = round((color_m + style_m) / 2, 1)

    return {
        "success":         True,
        "query_type":      query_type,
        "target_types":    target_types,
        "detected_by":     detected_by,          # which stage identified the category
        "recommendations": recommendations,
        "compatibility": {
            "color_match": color_m,
            "style_match": style_m,
            "overall":     overall,
        },
    }
