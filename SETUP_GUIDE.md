# Setup Guide

Reproduce the tested Windows setup using **PowerShell from the project folder**.
Commands use each environment's Python directly: activation and execution-policy
changes are unnecessary.

## Requirements

| Requirement | Details |
| --- | --- |
| OS | Windows x64; the integration was tested on Windows |
| Python | 3.13 for Flask/CLIP; 3.10 for diffusion; tested with 3.13.1 and 3.10.11 |
| GPU | Compatible NVIDIA driver/GPU; tested with a Quadro RTX 4000, 8 GB VRAM |
| Disk | Plan for roughly 40-50 GB free for environments, archive, models, data, and outputs; an estimate, not a measured minimum |
| Git | Required for source bootstrap; Git LFS is not needed by the downloader |
| Internet | Packages, initial assets, and CDN styles |
| Chrome | Only required for browser/screenshot tests |

Install Python from [python.org](https://www.python.org/downloads/windows/), Git
from [git-scm.com](https://git-scm.com/download/win), and a compatible NVIDIA
driver. PyTorch CUDA wheels include their runtime; a separate CUDA toolkit is
not part of the tested installation procedure. Upstream OOTDiffusion reports
Linux testing; these commands describe this project's local Windows integration.

## 1. Get the Project

Clone the repository. Authorized GitHub access is needed while it is private:

```powershell
git clone https://github.com/engr-abd-rehman/AI_Recommended_Outfit_-_VTON_System.git
Set-Location AI_Recommended_Outfit_-_VTON_System
py -0p
```

Both Python versions should be listed. If 3.10 is installed but not registered,
use its executable directly when creating `ootd_env`, for example:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv ootd_env
```

## 2. Main Environment

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
```

This runs Flask and CPU recommendations. CLIP ViT-B/32 loads lazily; its first
load downloads about 354 MB into the local CLIP cache if absent. No API key is
required.

## 3. Diffusion Environment

```powershell
py -3.10 -m venv ootd_env
.\ootd_env\Scripts\python.exe -m pip install --upgrade pip
.\ootd_env\Scripts\python.exe -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
.\ootd_env\Scripts\python.exe -m pip install -r ootd_requirements.txt
.\ootd_env\Scripts\python.exe -m pip install -r requirements-assets.txt
.\ootd_env\Scripts\python.exe -m pip check
.\ootd_env\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

CUDA should be `True`, runtime `11.8`. Do not combine environments: their Torch,
TorchVision, NumPy, and Pillow versions differ intentionally. CPU-only PyTorch
does not enable try-on because this implementation calls CUDA directly.

## 4. Download Upstream Source

```powershell
.\ootd_env\Scripts\python.exe scripts/setup_ootdiffusion.py
```

This clones the official repository, checks out
`13ef0faba266cdde9febc8ad39be2395bbb89d9c`, applies the documented OpenPose import
cleanup, and ensures `run/images_output/` exists. An already-applied patch is
recognized. A different existing revision is not reset or overwritten.
`OOTDiffusion/` is ignored runtime source, not a parent-repository submodule.
Its license is preserved; see [THIRD_PARTY.md](THIRD_PARTY.md).

## 5. Download the Data and Weights

**Review the provider's terms first.** The
[dataset agreement](https://github.com/yumingj/DeepFashion-MultiModal#agreement)
restricts use and redistribution. Do not push data or derived catalogs to GitHub.

```powershell
.\ootd_env\Scripts\python.exe scripts/restore_assets.py
```

| Source | Pinned revision or ID | Destination |
| --- | --- | --- |
| [Processed Text2Human archive](https://github.com/yumingj/Text2Human#1-dataset-preparation) | Drive ID `1KIoFfRZNQVn6RV_wTxG2wZmY8f2T_84B` | `.downloads/text2human-dataset.zip`, then `Kaggle_datasets/` |
| [OOTDiffusion checkpoints](https://huggingface.co/levihsu/OOTDiffusion) | `c79f9dd0585743bea82a39261cc09a24040bc4f9` | `OOTDiffusion/checkpoints/` |
| [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | `32bd64288804d66eefd0ccbe215aa642df71cc41` | `OOTDiffusion/checkpoints/clip-vit-large-patch14/` |

The archive is about **6.06 GB**, selected models **17.04 GB**. Model sizes and
SHA-256/Git-blob checksums are verified. `.part` transfers resume after failure.
ZIP extraction checks CRC and destination paths. CRC checks corruption, not
an independently published cryptographic signature for the dataset.

A fresh clone extracts training images, matching masks, and shape/texture
annotations without needing prebuilt metadata. If local metadata exists, its
filenames are used to select/verify the matching catalog. Existing vectors and
metadata are not overwritten by the downloader.

```powershell
.\ootd_env\Scripts\python.exe scripts/restore_assets.py --models
.\ootd_env\Scripts\python.exe scripts/restore_assets.py --dataset
.\ootd_env\Scripts\python.exe scripts/restore_assets.py --dataset --extract-only
```

The last command requires the complete archive already present. Run only one
restoration workflow against a destination at a time.

## 6. Build the Local Catalog

```powershell
.\venv\Scripts\python.exe build_index.py
```

This writes normalized float32, 512-dimensional `catalog_vectors.npy` and
`catalog_metadata.json` containing filenames, categories, garment types, and
annotations. The tested catalog contains **10,335 images**. CPU indexing can
take a significant amount of time. Rebuilding intentionally replaces both
files; do not rebuild a working catalog unnecessarily, and restart Flask after
changing the index so its cached vectors stay aligned.

## 7. Add Permitted Photos

Provide a clear front-facing, full-body photograph of one person as
`static/model/base_model.jpg` or `base_model.png`. A UI mannequin placeholder is
not an inference input. Use only photos you have permission to process.

For repeatable workflow checks, provide your own clean clothing photos as:

```text
static/Products/shirts/maroon-shirt.png
static/Products/pants/beige-pants.png
```

These are local fixture filenames, not bundled stock assets. Tests accept
`--shirt` and `--pants` paths for other filenames. Images, uploads, and generated
results remain ignored.

## 8. Start Flask

```powershell
.\venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5055 --no-reload
```

Open **http://127.0.0.1:5055**. Flask automatically launches the diffusion
environment. Choose a different free port if needed; do not stop unrelated
servers. Pass the same URL to browser checks using `--url`.

Try-on defaults: one sample, 40 steps per pass, guidance scale 2. Each outfit
uses an HD upper pass and DC lower pass. Submit one job at a time; HTTP 409
means another job is already running in that server process. Do not change
configuration while generating.

## 9. Verify and Capture

See [docs/TESTING.md](docs/TESTING.md) for all checks. Raw results and personal
screenshots stay local in ignored `docs/testing/` and `docs/screenshots/`.
With environments, permitted sample photos, and the server ready:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\ootd_env\Scripts\python.exe scripts/restore_and_test.py
```

This bootstraps missing source, restores assets, builds a missing index, checks
GPU/API behavior, and runs sequential browser generation tests. It does not
create environments or install packages. `--tests-only` skips downloads and
extraction but can build a missing index from already-present training data.
Keep the machine awake; do not submit interactive jobs during automated tests.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Python 3.10 launcher missing | Use the installed interpreter's full path |
| CUDA unavailable | NVIDIA driver, CUDA wheel versions, and correct environment |
| Index not ready | Extract data, then run `build_index.py` |
| Catalog HTTP 404 | Correct processed dataset, filenames, and `train_images` destination |
| Checkpoint missing | All 30 selected files, including separate CLIP encoder |
| Existing upstream revision differs | Preserve it; inspect a new `--target` directory instead of resetting |
| GPU memory failure or slow generation | One job/sample, actual logs, and GPU memory; fewer steps alone do not guarantee lower peak memory |
| HTTP 409 | Wait for the active job |
| Icons/layout missing | Bootstrap/Font Awesome CDN access |
| Port occupied | Choose a free port and update the browser-test URL |
| Share/properties incomplete | Documented prototype limitations, not setup failures |

Keep the development server on loopback. It has no production authentication or
service isolation. Read [SECURITY.md](SECURITY.md) before considering deployment.
