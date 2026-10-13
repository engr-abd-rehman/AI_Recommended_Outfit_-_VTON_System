"""
garment_cleaner.py  –  Upper-garment skin removal + inpainting for OOTDiffusion.

Given a catalog image + segmentation mask path, this module:
  1. Isolates the upper-garment region (segm label = 1).
  2. Detects non-garment, non-background pixels inside the garment
     bounding box (exposed skin, hands over shirt, arm occlusion).
  3. Inpaints those pixels with nearby garment texture (cv2 TELEA).
  4. Composites the cleaned garment on a pure-white 768 × 1024 canvas
     (OOTDiffusion cloth-image format) and saves as JPEG.

Results are cached: re-running the same image is instant.

Public API
----------
    clean_upper_garment(image_path, segm_dir, cache_dir) -> str
        Returns path to cleaned image, or original path on failure.

    score_garment_tryon_quality(image_path, segm_dir) -> float
        Returns 0.0–1.0 quality score for try-on suitability.
"""

import os
import numpy as np
import cv2
from PIL import Image

# ── OOTDiffusion canonical cloth-image size ────────────────────────────────────
OOTD_W, OOTD_H = 768, 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pair(image_path: str, segm_dir: str):
    """Load catalog image (RGB ndarray) + segm mask (2-D ndarray), or None."""
    stem      = os.path.splitext(os.path.basename(image_path))[0]
    segm_path = os.path.join(segm_dir, stem + "_segm.png")
    if not os.path.isfile(segm_path) or not os.path.isfile(image_path):
        return None, None
    img_rgb  = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    H, W     = img_rgb.shape[:2]
    segm_arr = np.array(
        Image.open(segm_path).resize((W, H), Image.NEAREST), dtype=np.uint8
    )
    return img_rgb, segm_arr


def _garment_bbox(garment_mask: np.ndarray, pad: int = 5):
    """Return (y1, y2, x1, x2) bounding box of True pixels, with padding."""
    H, W = garment_mask.shape
    rows = np.where(garment_mask.any(axis=1))[0]
    cols = np.where(garment_mask.any(axis=0))[0]
    y1, y2 = int(rows[0]) - pad, int(rows[-1]) + pad
    x1, x2 = int(cols[0]) - pad, int(cols[-1]) + pad
    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)


# ── Public: quality scorer ────────────────────────────────────────────────────

def score_garment_tryon_quality(image_path: str, segm_dir: str) -> float:
    """
    Score an upper-body catalog image for try-on source suitability.
    Returns 0.0 (bad) – 1.0 (ideal).

    Criteria (weighted):
      • Sleeve symmetry   (0.40) – garment spans both left & right halves
      • Occlusion ratio   (0.40) – non-garment, non-bg pixels inside bbox
      • Garment size      (0.20) – garment height / image height
    """
    img_rgb, segm_arr = _load_pair(image_path, segm_dir)
    if img_rgb is None:
        return 0.5

    garment_mask = (segm_arr == 1)
    if garment_mask.sum() < 500:
        return 0.1

    H, W = garment_mask.shape
    y1, y2, x1, x2 = _garment_bbox(garment_mask, pad=0)

    # Sleeve symmetry
    mid        = W // 2
    left_px    = garment_mask[:, :mid].sum()
    right_px   = garment_mask[:, mid:].sum()
    denom      = max(left_px, right_px, 1)
    symmetry   = min(left_px, right_px) / denom

    # Occlusion (non-garment, non-white pixels inside garment bbox)
    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True
    is_bg          = np.all(img_rgb > 220, axis=-1)
    occlusion_px   = int((bbox_mask & ~garment_mask & ~is_bg).sum())
    bbox_area      = max((y2 - y1) * (x2 - x1), 1)
    occ_ratio      = occlusion_px / bbox_area
    occ_score      = 1.0 - min(occ_ratio * 4.0, 1.0)   # 25 %+ → 0.0

    # Size coverage
    garment_h  = (y2 - y1) / H
    size_score = min(garment_h / 0.35, 1.0)             # 35 %+ height → 1.0

    quality = 0.40 * symmetry + 0.40 * occ_score + 0.20 * size_score
    return float(np.clip(quality, 0.0, 1.0))


# ── Public: cleaner ───────────────────────────────────────────────────────────

def clean_upper_garment(image_path: str, segm_dir: str, cache_dir: str) -> str:
    """
    Remove skin / occlusion pixels from an upper-body catalog image and
    return a clean 768 × 1024 white-background JPEG ready for OOTDiffusion.

    Returns the cached cleaned path on success, or the original path on
    failure / missing mask (graceful degradation).
    """
    stem       = os.path.splitext(os.path.basename(image_path))[0]
    cache_path = os.path.join(cache_dir, f"{stem}_ootd_clean.jpg")
    if os.path.isfile(cache_path):
        return cache_path

    img_rgb, segm_arr = _load_pair(image_path, segm_dir)
    if img_rgb is None:
        return image_path

    try:
        garment_mask = (segm_arr == 1)
        if garment_mask.sum() < 200:
            return image_path

        H, W = img_rgb.shape[:2]
        y1, y2, x1, x2 = _garment_bbox(garment_mask, pad=5)

        # Pixels inside bbox that are NOT garment AND NOT white-background
        bbox_mask  = np.zeros((H, W), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        is_bg      = np.all(img_rgb > 220, axis=-1)
        to_inpaint = (bbox_mask & ~garment_mask & ~is_bg).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if to_inpaint.sum() > 30:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inp_mask = cv2.dilate(to_inpaint * 255, kernel, iterations=1)
            img_bgr  = cv2.inpaint(img_bgr, inp_mask, 7, cv2.INPAINT_TELEA)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Composite garment on white background
        white      = np.full((H, W, 3), 255, dtype=np.uint8)
        mask3      = np.stack([garment_mask] * 3, axis=-1)
        composite  = np.where(mask3, img_rgb, white)

        # Crop to garment bbox, resize to fit 768 × 1024 canvas
        crop       = composite[y1:y2, x1:x2]
        cH, cW     = crop.shape[:2]
        scale      = min(OOTD_W / cW, OOTD_H / cH)
        nW, nH     = max(1, int(cW * scale)), max(1, int(cH * scale))
        resized    = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                (nW, nH), interpolation=cv2.INTER_LANCZOS4)
        canvas     = np.full((OOTD_H, OOTD_W, 3), 255, dtype=np.uint8)
        yo, xo     = (OOTD_H - nH) // 2, (OOTD_W - nW) // 2
        canvas[yo:yo+nH, xo:xo+nW] = resized

        os.makedirs(cache_dir, exist_ok=True)
        cv2.imwrite(cache_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[GarmentCleaner] Saved cleaned garment → {cache_path}", flush=True)
        return cache_path

    except Exception as exc:
        print(f"[GarmentCleaner] Error on {image_path}: {exc}", flush=True)
        return image_path
