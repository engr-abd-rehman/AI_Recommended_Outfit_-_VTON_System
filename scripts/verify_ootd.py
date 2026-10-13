"""Check the try-on environment without loading or downloading model weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"{checks[-1]['status']}: {name}: {detail}", flush=True)

    record("PyTorch NumPy integration", torch.from_numpy(np.array([1.0])).item() == 1.0,
           {"torch": torch.__version__, "numpy": np.__version__})
    available = torch.cuda.is_available()
    record("CUDA device available", available, torch.version.cuda)
    if available:
        result = torch.ones(1, device="cuda").sum().item()
        record("GPU tensor computation", result == 1.0, torch.cuda.get_device_name(0))

    for name, command, directory in (
        ("Try-on dependency consistency", [sys.executable, "-m", "pip", "check"], ROOT),
        ("OOTDiffusion inference imports", [sys.executable, "run_ootd.py", "--help"], ROOT / "OOTDiffusion" / "run"),
    ):
        process = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=90,
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        record(name, process.returncode == 0,
               {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})

    output = ROOT / "docs" / "testing" / "ootd-checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"python": sys.version, "checks": checks}, indent=2), encoding="utf-8")
    return int(any(check["status"] == "FAIL" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
