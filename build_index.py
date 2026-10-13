"""
build_index.py  –  Build CLIP catalog index from DeepFashion train_images.

Run once (takes several minutes on CPU):
    python build_index.py

Creates:
    catalog_vectors.npy      (N × 512  float32, L2-normalised)
    catalog_metadata.json    (list of N dicts with category, garment_type, labels)
"""

import os, sys, json, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import clip

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, "Kaggle_datasets", "train_images")
SHAPE_FILE  = os.path.join(BASE_DIR, "Kaggle_datasets", "shape_ann", "train_ann_file.txt")
TEXTURE_DIR = os.path.join(BASE_DIR, "Kaggle_datasets", "texture_ann", "train")
OUT_VEC     = os.path.join(BASE_DIR, "catalog_vectors.npy")
OUT_META    = os.path.join(BASE_DIR, "catalog_metadata.json")

BATCH_SIZE  = 64   # lower to 32 if you run out of RAM

# ── Category → garment type ────────────────────────────────────────────────────
GARMENT_TYPE = {
    "Tees_Tanks": "upper",          "Blouses_Shirts": "upper",
    "Sweaters": "upper",            "Sweatshirts_Hoodies": "upper",
    "Graphic_Tees": "upper",        "Shirts_Polos": "upper",
    "Denim": "lower",               "Pants": "lower",
    "Shorts": "lower",              "Skirts": "lower",
    "Leggings": "lower",            "Jackets_Coats": "outer",
    "Jackets_Vests": "outer",       "Cardigans": "outer",
    "Dresses": "full",              "Rompers_Jumpsuits": "full",
    "Suiting": "full",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_shape(path):
    ann = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) > 1:
                    ann[p[0]] = [int(x) for x in p[1:]]
    return ann

def _load_texture(tex_dir):
    ann = {}
    for part in ("upper", "lower", "outer"):
        fp = os.path.join(tex_dir, f"{part}_fused.txt")
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    ann.setdefault(p[0], {})[part] = int(p[1])
    return ann

def _parse_fn(fn):
    m = re.match(r"(MEN|WOMEN)-([^-]+)-", fn)
    return (m.group(1), m.group(2)) if m else ("UNK", "Unknown")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  CLIP Catalog Index Builder  –  DeepFashion train_images")
    print("=" * 60)

    print("\n[1/4] Loading annotations …")
    shape_ann   = _load_shape(SHAPE_FILE)
    texture_ann = _load_texture(TEXTURE_DIR)
    print(f"      shape={len(shape_ann)}  texture={len(texture_ann)}")

    print("\n[2/4] Loading CLIP ViT-B/32 …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"      device={device}")

    print("\n[3/4] Scanning train_images …")
    img_files = sorted([
        fn for fn in os.listdir(TRAIN_DIR)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    total = len(img_files)
    print(f"      {total} images found")

    print("\n[4/4] Extracting CLIP embeddings …")
    all_vecs = []
    all_meta = []
    errors   = 0

    for start in range(0, total, BATCH_SIZE):
        batch_fns = img_files[start : start + BATCH_SIZE]
        tensors, valid_fns = [], []

        for fn in batch_fns:
            try:
                img = preprocess(Image.open(os.path.join(TRAIN_DIR, fn)).convert("RGB"))
                tensors.append(img)
                valid_fns.append(fn)
            except Exception:
                errors += 1

        if not tensors:
            continue

        with torch.no_grad():
            t   = torch.stack(tensors).to(device)
            emb = model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalise
            all_vecs.append(emb.cpu().numpy().astype(np.float32))

        for fn in valid_fns:
            gender, cat = _parse_fn(fn)
            all_meta.append({
                "filename":     fn,
                "category":     cat,
                "garment_type": GARMENT_TYPE.get(cat, "unknown"),
                "gender":       gender,
                "shape_labels": shape_ann.get(fn, []),
                "texture":      texture_ann.get(fn, {}),
            })

        done = min(start + BATCH_SIZE, total)
        sys.stdout.write(f"\r      {done}/{total}  ({done/total*100:.1f}%)  errors={errors}")
        sys.stdout.flush()

    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    vecs = np.vstack(all_vecs)
    np.save(OUT_VEC, vecs)
    print(f"\n✅  Saved vectors  {vecs.shape}  →  {OUT_VEC}")

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"✅  Saved metadata {len(all_meta)} entries  →  {OUT_META}")
    print(f"\n   Skipped (corrupt/unreadable): {errors}")
    print("\n🚀  Now run:  python app.py")

if __name__ == "__main__":
    main()
