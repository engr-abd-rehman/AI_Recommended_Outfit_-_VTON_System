"""Run real application checks and write a report without changing the catalog."""

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import app as server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tryon-timeout", type=int, default=900)
    parser.add_argument("--skip-tryon", action="store_true", help="Leave expensive generation checks to browser tests")
    parser.add_argument("--shirt", type=Path, default=ROOT / "static/Products/shirts/maroon-shirt.png")
    parser.add_argument("--pants", type=Path, default=ROOT / "static/Products/pants/beige-pants.png")
    args = parser.parse_args()
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"{checks[-1]['status']}: {name}: {detail}", flush=True)

    client = server.app.test_client()
    for route in ("/", "/api/model-image", "/static/css/style.css", "/static/js/main.js"):
        response = client.get(route)
        record(f"GET {route}", response.status_code == 200, f"HTTP {response.status_code}")

    for route, payload, expected in (
        ("/upload", None, 400),
        ("/api/recommend", {}, 400),
        ("/api/recommend", {"filepath": "missing.png"}, 400),
        ("/tryon", {}, 400),
    ):
        response = client.post(route, json=payload)
        record(f"Validation {route} {payload}", response.status_code == expected, response.get_json())
    response = client.get("/tryon/progress/unknown-job")
    record("Unknown try-on job", response.status_code == 404, response.get_json())

    for filename, content in (("invalid.txt", b"not an image"), ("invalid.png", b"not an image")):
        response = client.post("/upload", data={"file": (io.BytesIO(content), filename)})
        record(f"Reject {filename}", response.status_code == 400, response.get_json())
        if response.status_code == 200:
            Path(response.get_json()["filepath"]).unlink(missing_ok=True)

    vectors = np.load(ROOT / "catalog_vectors.npy")
    metadata = json.loads((ROOT / "catalog_metadata.json").read_text(encoding="utf-8"))
    record("Catalog vector alignment", vectors.shape == (len(metadata), 512) and np.isfinite(vectors).all(),
           {"vector_shape": list(vectors.shape), "metadata_items": len(metadata),
            "finite": bool(np.isfinite(vectors).all())})
    available = sum((Path(server.CATALOG_DIR) / item["filename"]).is_file() for item in metadata)
    record("Catalog source images", available == len(metadata),
           {"available": available, "required": len(metadata)})
    record("OOTDiffusion environment", Path(server.OOTD_PYTHON).is_file(), server.OOTD_PYTHON)
    checkpoint_dir = ROOT / "OOTDiffusion" / "checkpoints"
    manifest_file = ROOT / ".downloads" / "model-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else []
    complete = len(manifest) == 30 and all((ROOT / item["path"]).is_file()
                                         and (ROOT / item["path"]).stat().st_size == item["size"] for item in manifest)
    record("OOTDiffusion checkpoints", complete,
           {"directory": str(checkpoint_dir), "verified_manifest_files": len(manifest), "all_sizes_match": complete})

    samples = ((args.shirt, "upper", "lower"), (args.pants, "lower", "upper"))
    for sample, expected_type, target_type in samples:
        with sample.open("rb") as file:
            response = client.post("/upload", data={"file": (file, sample.name)})
        uploaded = response.get_json()
        record(f"Upload {sample.name}", response.status_code == 200 and uploaded.get("success"), uploaded)
        if not uploaded.get("success"):
            continue
        started = time.monotonic()
        response = client.post("/api/recommend", json={"filepath": uploaded["filepath"]})
        result = response.get_json() or {}
        recs = result.get("recommendations", [])
        record(f"Recommend {sample.name}",
               response.status_code == 200 and result.get("success") and len(recs) == 4
               and result.get("query_type") == expected_type
               and all(item["garment_type"] == target_type for item in recs),
               {"seconds": round(time.monotonic() - started, 2), "http_status": response.status_code,
                "result": result})
        for item in recs:
            image_response = client.get(item["image_url"])
            record(f"Recommendation image {item['filename']}", image_response.status_code == 200,
                   f"HTTP {image_response.status_code}")
        if recs and not args.skip_tryon:
            response = client.post("/tryon", json={"uploaded_filepath": uploaded["filepath"],
                                                   "catalog_filename": recs[0]["filename"],
                                                   "uploaded_type": result.get("query_type")})
            job = response.get_json() or {}
            if job.get("job_id"):
                job_id = job["job_id"]
                deadline = time.monotonic() + args.tryon_timeout
                while True:
                    job = client.get(f"/tryon/progress/{job_id}").get_json()
                    if job.get("status") != "running":
                        break
                    if time.monotonic() >= deadline:
                        job["error"] = f"Generation exceeded {args.tryon_timeout} seconds"
                        break
                    time.sleep(2)
            record(f"Try-on {sample.name}", job.get("status") == "done", job)

    directory = ROOT / "docs" / "testing"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "checks": checks,
              "passed": sum(item["status"] == "PASS" for item in checks),
              "failed": sum(item["status"] == "FAIL" for item in checks)}
    (directory / "system-checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {directory / 'system-checks.json'}", flush=True)
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
