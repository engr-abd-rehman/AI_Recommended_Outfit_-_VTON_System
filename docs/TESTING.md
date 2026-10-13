# Testing and Known Limitations

## Observed Locally

Verified on 13 October 2026 with Python 3.13.1/3.10.11 and a Quadro RTX 4000,
8 GB VRAM. This records evidence, not a guarantee for every input or machine.

| Check | Evidence |
| --- | --- |
| Declared environments | Installed; `pip check` passed in both |
| GPU environment | 5 checks passed, including a real CUDA tensor and inference imports |
| Restored models | All 30 selected model files verified against published sizes/checksums |
| Restored catalog | 10,335 indexed originals and matching segmentation masks present |
| API/catalog workflows | 27 checks passed with duplicate generation skipped |
| Fresh upper-upload generation | HD + DC completed; new PNG rendered and downloaded successfully |
| Preview timing regression | 4 real Chrome upload/replacement checks passed |
| Restoration safety | 7 offline tests passed, including fresh-clone extraction without metadata |
| Job admission | 3 isolated tests passed for busy rejection, completion, and failed-worker recovery |
| Fresh upstream bootstrap | Pinned source cloned, patch applied, inference `--help` passed |

The subsequent full browser retest was stopped when it overlapped with an
interactive job and overloaded the GPU. A single-job guard was added after that
observation. A newly generated lower-upload outfit has not yet been verified in
this publication review. Do not call the entire system perfect or fully tested.

## Offline and CI Checks

```powershell
py -3.10 -m venv .test_env
.\.test_env\Scripts\python.exe -m pip install -r requirements-test.txt
.\.test_env\Scripts\python.exe scripts/test_restore_assets.py
.\.test_env\Scripts\python.exe scripts/test_job_guard.py
.\.test_env\Scripts\python.exe scripts/audit_publication.py --check-only
```

The GitHub workflow uses this lightweight dependency set. It does not download
restricted data or 17 GB of weights, and it does not test actual CUDA inference.

## GPU and Real API Checks

After completing setup and supplying permitted local photos:

```powershell
.\ootd_env\Scripts\python.exe scripts/verify_ootd.py
.\venv\Scripts\python.exe scripts/verify_system.py --skip-tryon
```

Without `--skip-tryon`, the API script performs two actual sequential generations.
Do not run it alongside browser generation tests or interactive requests.
Use `--shirt` and `--pants` to supply alternative local file paths.

## Browser and Private Screenshots

Keep Flask running on the chosen loopback port:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe scripts/test_upload_preview.py --url http://127.0.0.1:5055
.\venv\Scripts\python.exe scripts/capture_screenshots.py --url http://127.0.0.1:5055
```

The browser test waits for each generation to finish, checks the rendered result
URL and download response, and saves new images only after successful inference.
Reports are updated incrementally, including when automation aborts. Use
`--garment upper` or `--garment lower` for one category, `--tryon-timeout` for the
generation wait, and `--shirt`/`--pants` for your permitted fixtures.

Local reports are under `docs/testing/`; data-bearing captures under
`docs/screenshots/`. Both are ignored because they can contain personal images,
restricted catalog previews, and machine paths. Never blindly add them to Git.

## Publication-Safe Captures

```powershell
.\venv\Scripts\python.exe scripts/capture_readme_screenshots.py
```

This launches an isolated clean preview, captures desktop/phone layouts into
`docs/images/`, then stops only its own server. These screenshots show the real
interface without a supplied photo or downloaded catalog. They are not fabricated
generation results. Adding real result screenshots requires a separate rights
review for every input and any visible catalog material.

## Remaining Limitations

- Garment shape, sleeves, length, color, and details can change during diffusion.
  An observed result turned recommended shorts into trousers and lengthened sleeves.
- Match percentages are heuristic UI scores, not calibrated probabilities or accuracy.
- The detected panel currently populates category; other properties are not reliably populated.
- Share has no working handler; view controls are not a multi-angle generation engine.
- Only the shirt/pants two-pass workflow has the stated inference evidence.
- Intermediate progress can include model-loading bars; it is not a precise ETA.
- Jobs are in memory; shared intermediate/output names rely on one active job per process.
- Multi-process concurrency, cancellation, rate limiting, authentication, and public deployment are not implemented.
- Input diversity, fairness, adversarial uploads, extensive visual fidelity, and performance benchmarks remain unverified.

See [SECURITY.md](../SECURITY.md) for deployment boundaries and
[THIRD_PARTY.md](../THIRD_PARTY.md) for asset terms.
