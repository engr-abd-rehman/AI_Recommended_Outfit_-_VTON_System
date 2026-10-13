"""Run resumable asset restoration followed by sequential real system tests."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "testing"
STATUS = REPORTS / "restoration-status.json"
OOTD_PYTHON = ROOT / "ootd_env" / "Scripts" / "python.exe"
APP_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
MODEL_BYTES = 17043500000
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def disk_bytes(paths):
    total = 0
    for path in paths:
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Completed downloads are atomically renamed while progress is sampled.
            pass
    return total


def save_status(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_dir = ROOT / "OOTDiffusion" / "checkpoints"
    state["model_downloaded_bytes"] = disk_bytes(checkpoint_dir.rglob("*"))
    state["model_expected_bytes_approx"] = MODEL_BYTES
    state["dataset_downloaded_bytes"] = disk_bytes((ROOT / ".downloads").glob("text2human-dataset.zip*"))
    state["dataset_expected_bytes_approx"] = 6060000000
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def summarize():
    date = datetime.now(timezone(timedelta(hours=5))).strftime("%d %B %Y")
    lines = ["# Asset Restoration and Retest", "", f"Date: {date} (Asia/Karachi).", "",
             "Assets were restored from pinned official model revisions and the processed Text2Human dataset.",
             "Large model files were verified against their published SHA-256 checksums; extracted ZIP members passed CRC verification.", ""]
    for filename, title in (("ootd-checks.json", "GPU environment"), ("system-checks.json", "API and catalog"),
                            ("browser-checks.json", "Browser and fresh generation")):
        report = json.loads((REPORTS / filename).read_text(encoding="utf-8"))
        checks = report["checks"]
        failures = [item for item in checks if item["status"] == "FAIL"]
        lines.append(f"- {title}: {len(checks) - len(failures)} passed, {len(failures)} failed. [{filename}]({filename})")
        for item in failures:
            lines.append(f"  Failure: {item['name']}: {item['detail']}")
    lines.extend(["", "API checks skip expensive try-on requests; the browser performs them sequentially for both upload categories.",
                  "No cached result is substituted for a newly generated outfit.", "",
                  "Passing inference checks does not guarantee faithful garment reproduction. An earlier fresh upper-upload output changed sleeve length and turned recommended shorts into trousers.", "",
                  "Screenshots: [gallery](../screenshots/README.md).", "",
                  "Known non-inference limitations: cosmetic match percentages, incomplete detected properties, unimplemented Share control, and untested concurrent jobs.",
                  "The Flask development server is local only, not a production deployment.", ""])
    (REPORTS / "RESTORATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    browser = json.loads((REPORTS / "browser-checks.json").read_text(encoding="utf-8"))
    gallery = ["# Application Screenshots", "", f"Captured from the live application on {date} using installed Chrome.", "",
               "## Desktop", "", "![Application home](01-home-desktop.png)", "",
               "## Recommendations", "", "![Shirt recommendations](03-recommendations-upper.png)", "",
               "![Pants recommendations](03-recommendations-lower.png)", ""]
    for label in ("upper", "lower"):
        completed = any(item["name"] == f"Completed generation {label}" for item in browser["checks"])
        if not completed and browser.get("phase") != "complete":
            gallery.extend([f"## {label.title()} Upload: Retest Incomplete", "",
                            "The browser test stopped before this result could be verified. Older captures are not presented as current results.", ""])
            continue
        success = any(item["name"] == f"Completed generation {label}" and item["status"] == "PASS" for item in browser["checks"])
        heading = "Fresh Generated Result" if success else "Try-On Attempt (Failed)"
        gallery.extend([f"## {heading}: {label.title()} Upload", "", f"![Try-on {label}](04-tryon-{label}.png)", ""])
        if success:
            gallery.extend([f"![New generated outfit {label}](06-generated-outfit-{label}.png)", ""])
    gallery.extend(["## Phone", "", "![Phone layout](05-home-mobile.png)", "",
                    "The homepage shows the supplied base model, not a generation. Match percentages are heuristic presentation values, not measured accuracy.",
                    "See [retest report](../testing/RESTORATION_REPORT.md).", ""])
    (ROOT / "docs" / "screenshots" / "README.md").write_text("\n".join(gallery), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5055")
    parser.add_argument("--tests-only", action="store_true", help="Retest restored assets without repeating downloads or extraction")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    state = {"phase": "testing" if args.tests_only else "downloading", "url": args.url,
             "started_at": datetime.now(timezone.utc).isoformat(), "exit_codes": {}}
    save_status(state)
    if not args.tests_only:
        if not (ROOT / "OOTDiffusion" / "run" / "run_ootd.py").is_file():
            subprocess.run([str(OOTD_PYTHON), "scripts/setup_ootdiffusion.py"], cwd=ROOT,
                           env=CHILD_ENV, check=True)
        commands = (("models", [str(OOTD_PYTHON), "-u", "scripts/restore_assets.py", "--models"]),
                    ("dataset", [str(OOTD_PYTHON), "-u", "scripts/restore_assets.py", "--dataset"]))
        processes = {name: subprocess.Popen(command, cwd=ROOT, env=CHILD_ENV) for name, command in commands}
        while any(process.poll() is None for process in processes.values()):
            state["exit_codes"] = {name: process.poll() for name, process in processes.items()}
            save_status(state)
            print(f"Assets: models {state['model_downloaded_bytes'] / 1e9:.2f} GB, dataset {state['dataset_downloaded_bytes'] / 1e9:.2f} GB", flush=True)
            time.sleep(30)
        state["exit_codes"] = {name: process.returncode for name, process in processes.items()}
        if any(process.returncode != 0 for process in processes.values()):
            state["phase"] = "restoration_failed"
            save_status(state)
            return 1
    if not (ROOT / "catalog_metadata.json").is_file() or not (ROOT / "catalog_vectors.npy").is_file():
        state["phase"] = "building_index"
        save_status(state)
        state["exit_codes"]["index"] = subprocess.run([str(APP_PYTHON), "-u", "build_index.py"],
                                                       cwd=ROOT, env=CHILD_ENV).returncode
        if state["exit_codes"]["index"] != 0:
            state["phase"] = "index_failed"
            save_status(state)
            return 1
    state["phase"] = "testing"
    state["assets_restored"] = True
    save_status(state)
    try:
        with urllib.request.urlopen(args.url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError("Application homepage is unavailable")
        for name, command in (
            ("gpu", [str(OOTD_PYTHON), "-u", "scripts/verify_ootd.py"]),
            ("api", [str(APP_PYTHON), "-u", "scripts/verify_system.py", "--skip-tryon"]),
            ("browser", [str(APP_PYTHON), "-u", "scripts/capture_screenshots.py", "--url", args.url]),
        ):
            state["active_test"] = name
            save_status(state)
            state["exit_codes"][name] = subprocess.run(command, cwd=ROOT, env=CHILD_ENV).returncode
            save_status(state)
        summarize()
    except Exception as error:
        state["phase"] = "tests_failed"
        state["error"] = str(error)
        save_status(state)
        raise
    state.pop("active_test", None)
    state["phase"] = "complete" if all(code == 0 for code in state["exit_codes"].values()) else "tests_failed"
    save_status(state)
    print(f"Workflow finished: {state['phase']}. See docs/testing/RESTORATION_REPORT.md", flush=True)
    return int(state["phase"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
