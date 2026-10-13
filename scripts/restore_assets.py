"""Restore pinned official weights and extract the matching public dataset.

Run with ootd_env's Python. Dataset downloads use public links without cookies.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import zipfile

import gdown
from huggingface_hub import HfApi
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".downloads"
CHECKPOINTS = ROOT / "OOTDiffusion" / "checkpoints"
DATASET_ID = "1KIoFfRZNQVn6RV_wTxG2wZmY8f2T_84B"
REPOSITORIES = (
    ("levihsu/OOTDiffusion", "c79f9dd0585743bea82a39261cc09a24040bc4f9"),
    ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
)
CLIP_FILES = {"config.json", "preprocessor_config.json", "model.safetensors",
              "merges.txt", "special_tokens_map.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json"}


def digest(path, algorithm="sha256", git_blob=False):
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download_models(workers):
    api = HfApi()
    tasks = []
    for repo, revision in REPOSITORIES:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        for entry in info.siblings:
            name = entry.rfilename
            if repo == "levihsu/OOTDiffusion":
                if not name.startswith("checkpoints/") or name.endswith(".pth") and "humanparsing" in name:
                    continue
                relative = PurePosixPath(name).relative_to("checkpoints")
            else:
                if name not in CLIP_FILES:
                    continue
                relative = PurePosixPath("clip-vit-large-patch14") / name
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe checkpoint path: {relative}")
            tasks.append({"repo": repo, "revision": revision, "remote": name,
                          "path": CHECKPOINTS.joinpath(*relative.parts), "size": entry.size,
                          "sha256": entry.lfs.get("sha256") if entry.lfs else None,
                          "blob_id": entry.blob_id})

    progress = {}
    lock = threading.Lock()

    def verified(path, item):
        if not path.is_file() or path.stat().st_size != item["size"]:
            return False
        if item["sha256"]:
            return digest(path) == item["sha256"]
        return digest(path, "sha1", git_blob=True) == item["blob_id"]

    def transfer(item):
        path = item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        if verified(path, item):
            print(f"Verified existing: {path.relative_to(ROOT)}", flush=True)
            return
        # Large CDN transfers can be interrupted repeatedly; preserve each partial.
        for attempt in range(50):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["size"]:
                    raise ValueError(f"Partial file exceeds expected size: {partial}")
                if offset < item["size"]:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    url = (f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/"
                           f"{item['remote']}?download=true&restore={int(time.time())}")
                    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if offset and response.status_code == 206:
                            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise ValueError("Server returned an unexpected resume offset")
                            mode = "ab"
                        else:
                            offset = 0
                            mode = "wb"
                        with partial.open(mode) as file:
                            for block in response.iter_content(1024 * 1024):
                                file.write(block)
                                offset += len(block)
                                if offset > item["size"]:
                                    raise ValueError("Download exceeds published file size")
                                with lock:
                                    progress[str(path)] = offset
                if not verified(partial, item):
                    partial.unlink(missing_ok=True)
                    raise ValueError("Downloaded file failed published checksum verification")
                partial.replace(path)
                print(f"Verified download: {path.relative_to(ROOT)}", flush=True)
                return
            except (requests.RequestException, ValueError) as error:
                # Do not print redirect URLs, which can contain signed access tokens.
                print(f"Retry {attempt + 1}/50: {path.relative_to(ROOT)} ({type(error).__name__})", flush=True)
                if attempt == 49:
                    raise RuntimeError(f"Could not download {item['remote']}: {type(error).__name__}") from None
                time.sleep(min(2 ** attempt, 30))

    total = sum(item["size"] for item in tasks)
    print(f"Restoring {len(tasks)} model files, {total / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transfer, item) for item in tasks}
        while pending:
            done, pending = wait(pending, timeout=30)
            for future in done:
                future.result()
            complete = sum(item["path"].stat().st_size if item["path"].is_file()
                           else progress.get(str(item["path"]), 0) for item in tasks)
            print(f"Model progress: {complete / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    manifest = [{**item, "path": str(item["path"].relative_to(ROOT))} for item in tasks]
    (CACHE / "model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def restore_dataset(extract_only):
    archive = CACHE / "text2human-dataset.zip"
    if not extract_only:
        print("Downloading official processed Text2Human dataset (about 6.06 GB)", flush=True)
        result = gdown.download(id=DATASET_ID, output=str(archive), use_cookies=False,
                                resume=True, retries=20, timeout=60, quiet=True)
        if result is None:
            raise RuntimeError("Official dataset download failed")
    metadata_path = ROOT / "catalog_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    filenames = {item["filename"] for item in metadata} if metadata is not None else None
    destination = ROOT / "Kaggle_datasets"
    selected = []
    with zipfile.ZipFile(archive) as bundle:
        members = []
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if ".." in parts or PurePosixPath(entry.filename).is_absolute():
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            for index, part in enumerate(parts):
                if part in {"train_images", "segm", "shape_ann", "texture_ann"}:
                    relative = PurePosixPath(*parts[index:])
                    members.append((entry, relative))
                    break
        if filenames is None:
            filenames = {relative.name for _, relative in members if relative.parts[0] == "train_images"
                         and relative.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        if not filenames:
            raise ValueError("Archive contains no required training images")
        masks = {PurePosixPath(name).stem + "_segm.png" for name in filenames}
        selected = [(entry, relative) for entry, relative in members
                    if relative.parts[0] not in {"train_images", "segm"}
                    or relative.parts[0] == "train_images" and relative.name in filenames
                    or relative.parts[0] == "segm" and relative.name in masks]
        available = {relative.name for _, relative in selected if relative.parts[0] == "train_images"}
        missing = filenames - available
        if missing:
            raise ValueError(f"Archive does not match catalog: {len(missing)} missing images")
        missing_masks = masks - {relative.name for _, relative in selected if relative.parts[0] == "segm"}
        if missing_masks:
            raise ValueError(f"Dataset is missing {len(missing_masks)} required segmentation masks")
        print(f"Extracting {len(selected)} required dataset files; archive CRC checked while reading", flush=True)
        for count, (entry, relative) in enumerate(selected, 1):
            target = destination.joinpath(*relative.parts)
            if destination.resolve() not in target.resolve().parents:
                raise ValueError(f"Archive member escapes destination: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with bundle.open(entry) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)
            if count % 1000 == 0:
                print(f"Extracted {count}/{len(selected)}", flush=True)
    report = {"source": "https://github.com/yumingj/Text2Human", "drive_id": DATASET_ID,
              "archive_bytes": archive.stat().st_size, "catalog_images": len(available),
              "missing_masks": sorted(missing_masks), "extracted_files": len(selected)}
    (CACHE / "dataset-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Dataset restored: {len(available)} catalog images and matching masks", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    if args.models or not args.dataset:
        download_models(args.workers)
    if args.dataset or not args.models:
        restore_dataset(args.extract_only)


if __name__ == "__main__":
    main()
