# AI Recommended Outfit & VTON System

**FashionAI | Local outfit discovery and two-pass virtual try-on**

[Source Checks](https://github.com/engr-abd-rehman/AI_Recommended_Outfit_-_VTON_System/actions/workflows/checks.yml)
[![Original Code: MIT](https://img.shields.io/badge/original_code-MIT-green)](LICENSE)

A desktop-first research application that connects clothing-image recommendations
to outfit generation. Upload a shirt or pair of pants, compare four complementary
catalog options, select a look, and generate a new image on a supplied model
photograph. Recommendations use local CLIP embeddings; try-on uses separate
upper- and lower-body OOTDiffusion passes.

This repository contains the application, reproducible setup scripts, tests, and
documentation. It does **not** contain datasets, model weights, personal photos,
or prebuilt catalog indexes. The complete installation procedure is included
in this README, including download sources, sizes, and destinations.

> **Project status:** a locally tested research prototype, not a production
> fitting service. It estimates visual compatibility and generates images;
> it does not measure garment fit, sizing, or recommendation accuracy.

## Contents

- [Application Preview](#application-preview)
- [Capabilities](#capabilities)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Using the Application](#using-the-application)
- [Repository Structure](#repository-structure)
- [File Responsibilities](#file-responsibilities)
- [HTTP API](#http-api)
- [Testing and Verification](#testing-and-verification)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Security and Publication](#security-and-publication)
- [Contributing](#contributing)
- [License and Attribution](#license-and-attribution)

## Application Preview

![FashionAI desktop workspace: upload, detected properties, recommendations area, and virtual try-on panel](docs/images/interface-desktop.png)

**Desktop workspace.** The left side contains clothing upload, detected category,
and outfit recommendations. The right side displays the supplied model photograph,
generation progress, and final downloadable result.

This is a real application capture from an isolated source-only workspace.
The model area is a placeholder because no permitted photo or catalog was supplied
to that preview. It is **not** a recommendation or generated-outfit result.
Data-bearing screenshots stay local until all visible and contributing assets
have been cleared for redistribution.

## Capabilities

| Area | Implemented behavior |
| --- | --- |
| Image upload | File selection and drag-and-drop; verifies image content before saving |
| Category detection | Upper/lower detection using local metadata and image/CLIP heuristics |
| Recommendations | Four complementary-category results with ranking and deduplication |
| Garment previews | Segmentation-based cutouts from locally downloaded originals and masks |
| Outfit selection | Choose a catalog garment to combine with the uploaded garment |
| Virtual try-on | OOTDiffusion HD upper-body pass followed by DC lower-body pass |
| Background processing | Job ID, progress polling, success/error states, one active job per process |
| Export | Download the generated PNG |
| Asset restoration | Pinned sources, resumable model transfers, verified hashes, safe ZIP extraction |
| Verification | Offline safety tests, CUDA checks, real API tests, browser captures |

Match percentages are **heuristic presentation scores**, not calibrated
probabilities. Category is the established properties-panel output; other
displayed attributes are not reliably implemented. Share and view controls do
not imply working social sharing or multi-angle generation.

## Technology Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Server | Flask / Jinja | Local endpoints and application page |
| Interface | HTML, CSS, JavaScript | Uploads, selection, progress, export |
| UI dependencies | Bootstrap, Font Awesome | CDN-loaded layout and icons |
| Recommendation encoder | OpenAI CLIP ViT-B/32 | Normalized 512-dimensional image embeddings |
| Catalog search | NumPy | Similarity, category filtering, ranking |
| Image processing | Pillow, OpenCV | Validation, masks, crops, garment preparation |
| Generation | OOTDiffusion, Diffusers, Transformers | Upper-body HD and lower-body DC inference |
| GPU execution | PyTorch CUDA 11.8 | Separate Python 3.10 inference stack |
| Automation | Playwright / Chrome | Browser regressions and workflow capture |
| CI | GitHub Actions | Source and offline safety without data or GPU |

**Two environments are intentional.** Flask/recommendations use Python 3.13 and
CPU PyTorch; diffusion uses Python 3.10 and its CUDA-compatible model stack.
Their Torch, NumPy, and Pillow versions should not be combined.

## Architecture

```text
Browser
  |
  +-- Upload --> Flask image validation --> local upload
  |
  +-- Recommend --> CLIP embedding + local catalog
  |                 + complementary-category filtering
  |                 + quality ranking and deduplication
  |              --> four garment previews
  |
  +-- Select look + Try On --> single-job admission
                               |
                               +-- prepare shirt and pants
                               +-- HD: shirt onto base model photograph
                               +-- DC: pants onto HD output
                               +-- unique result PNG
                               |
Browser <-- progress / result URL <-- in-memory job registry
```

The catalog is built locally from processed training images and annotations.
The tested index has **10,335 rows x 512 dimensions**, stored as normalized
float32 vectors alongside aligned metadata. CLIP loads lazily on the first request.

A background thread launches the inference interpreter explicitly. The second
pass receives the first pass's result, not the original model photo. The final
DC output is used directly; the historical compositor helper is not part of the
active output path.

## Installation

Run the following commands in **PowerShell from the project directory**.
Each environment's Python is invoked directly, so activation and execution-policy
changes are unnecessary.

### Prerequisites

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

### 1. Get the Project

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

### 2. Main Environment

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

### 3. Diffusion Environment

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

### 4. Download Upstream Source

```powershell
.\ootd_env\Scripts\python.exe scripts/setup_ootdiffusion.py
```

This clones the official repository, checks out
`13ef0faba266cdde9febc8ad39be2395bbb89d9c`, applies the documented OpenPose import
cleanup, and ensures `run/images_output/` exists. An already-applied patch is
recognized. A different existing revision is not reset or overwritten.
`OOTDiffusion/` is ignored runtime source, not a parent-repository submodule.
Its license is preserved; see [THIRD_PARTY.md](THIRD_PARTY.md).

### 5. Download the Data and Weights

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

### 6. Build the Local Catalog

```powershell
.\venv\Scripts\python.exe build_index.py
```

This writes normalized float32, 512-dimensional `catalog_vectors.npy` and
`catalog_metadata.json` containing filenames, categories, garment types, and
annotations. The tested catalog contains **10,335 images**. CPU indexing can
take a significant amount of time. Rebuilding intentionally replaces both
files; do not rebuild a working catalog unnecessarily, and restart Flask after
changing the index so its cached vectors stay aligned.

### 7. Add Permitted Photos

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

### 8. Start Flask

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

### 9. Verify and Capture

See [Testing and Verification](#testing-and-verification) below for the checks. Raw results and personal
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


## Using the Application

### Desktop Workflow

1. Start Flask using [the server command](#8-start-flask) and open
   `http://127.0.0.1:5055`.
2. Upload a clear shirt or pants photograph. Wait for the correct image preview.
3. Select **Get Recommendations**. The first request may take longer while CLIP
   initializes or downloads its cached weights.
4. Compare the four complementary garments. A shirt upload searches lower-body
   options; a pants upload searches upper-body options.
5. Choose **Select Look**, then **Try On Outfit**.
6. Leave the page and server running while both inference passes complete.
7. Review the output, then use **Download** to retrieve the PNG.

Try-on uses the configured photograph in `static/model/`, not a webcam or body
scan. Changing an upload clears stale recommendation selection.

### Input Guidance and Runtime

| Input | Useful characteristics | Avoid |
| --- | --- | --- |
| Model photo | One person, front-facing, full body, clear limbs | Crowds, extreme poses, cropped body, uncertain consent |
| Garment photo | Clear outline, simple background, visible texture/color | Heavy occlusion, collages, damaged images |
| Catalog and masks | Matching filenames, aligned segmentation | Mixed archives, missing masks, stale index |
| Generated result | Review color, sleeve length, hem, shape, artifacts | Treating a plausible image as proof of fit |

- Recommendations run on CPU; diffusion needs the configured NVIDIA GPU.
- Default generation uses one sample, 40 steps per pass, guidance scale 2.
  An outfit can take several minutes or longer on the tested GPU.
- Progress is approximate and includes loading messages, not a reliable ETA.
- Submit one generation at a time; HTTP `409` means that process is busy.
- Restarting Flask loses in-memory jobs. Keep the machine awake during inference.
- No OpenAI API key, Kaggle account, or paid inference API is required.
  Initial downloads and CDN styles still require network access.

## Repository Structure

```text
AI_Recommended_Outfit_-_VTON_System/
|-- README.md                       # installation, usage, engineering overview
|-- app.py                          # Flask application and inference dispatch
|-- recommender.py                  # CLIP recommendation engine
|-- build_index.py                  # local catalog builder
|-- garment_cleaner.py              # scoring and optional cleaning
|-- requirements*.txt               # application, assets, browser, test pins
|-- ootd_requirements.txt            # separate diffusion stack
|-- templates/index.html
|-- static/
|   |-- css/style.css
|   |-- js/main.js
|   |-- model/.gitkeep               # add permitted photo locally
|   `-- Products/                   # local test-photo placeholders
|-- scripts/                        # bootstrap, restoration, tests, captures
|-- patches/ootdiffusion-local.patch
|-- licenses/                       # upstream license copy
|-- docs/
|   |-- images/                     # reviewed publication images
|   |-- ARCHITECTURE.md
|   |-- TESTING.md
|   `-- PUBLISHING.md
|-- .github/workflows/checks.yml
|-- publication-files.json          # publication allowlist
|-- THIRD_PARTY.md
|-- LICENSE
|-- LICENSE_POLICY.md
`-- SECURITY.md
```

After setup, these ignored local resources also exist:

| Location | Purpose | Repository payload? |
| --- | --- | --- |
| `venv/` | Flask, CPU CLIP, indexing | No |
| `ootd_env/` | CUDA inference and asset tooling | No |
| `OOTDiffusion/` | Upstream source and checkpoints | No |
| `Kaggle_datasets/` | Training originals, masks, annotations | No |
| `catalog_vectors.npy` / `catalog_metadata.json` | Local vectors and aligned metadata | No |
| `static/model/` / `static/Products/` | Permitted operator-supplied photos | Placeholders only |
| `static/uploads/` / `static/crops/` | Inputs, intermediate/results, cached crops | No |
| `.downloads/` | Archives, logs, manifests | No |
| `docs/testing/` / `docs/screenshots/` | Raw reports and data-bearing captures | No |

## File Responsibilities

### Application

| File | Responsibility |
| --- | --- |
| `app.py` | Routes, uploads, image validation, mask crops, inference preparation, two-pass dispatch, job/progress state |
| `recommender.py` | Lazy CLIP/index loading, category detection, complementary search, quality ranking, deduplication |
| `build_index.py` | Sorted scan, annotation parsing, batch encoding, normalization, vectors and metadata |
| `garment_cleaner.py` | Try-on suitability heuristics and optional cleaning utilities |
| `templates/index.html` | Upload, properties, recommendations, model/result panels |
| `static/js/main.js` | File/drop input, preview, API calls, selection, polling, downloads |
| `static/css/style.css` | Desktop layout, garment cards, loading/result states, viewport breakpoints |

### Scripts

Paths below are relative to `scripts/`.

| Script | Responsibility |
| --- | --- |
| `setup_ootdiffusion.py` | Pin official source, apply documented patch; never reset a different checkout |
| `restore_assets.py` | Downloads, resume, hashes/sizes, safe extraction |
| `restore_and_test.py` | Coordinate source/assets, missing index, checks, private captures |
| `verify_ootd.py` | CUDA, real tensor operation, dependencies, inference imports |
| `verify_system.py` | Actual uploads, recommendations, catalog/model routes, optional inference |
| `test_restore_assets.py` | Archive safety, fresh extraction, hashing, missing assets, transfer resume |
| `test_job_guard.py` | Busy rejection, completion, failed worker-start recovery |
| `test_upload_preview.py` | Browser regressions for replacing uploads |
| `capture_screenshots.py` | Real recommendation/inference workflows; captures private by default |
| `capture_readme_screenshots.py` | Isolated source-only desktop capture without data/photos |
| `audit_publication.py` | Allowlist, exclusions, size/link/content checks, optional exact release copy |

### Configuration and Governance

| File | Responsibility |
| --- | --- |
| `requirements.txt` | Main application package pins |
| `ootd_requirements.txt` | Inference package pins |
| `requirements-assets.txt` | Small download-tool dependency set |
| `requirements-dev.txt` | Browser automation dependency |
| `requirements-test.txt` | Offline/CI dependencies without Torch or model weights |
| `.github/workflows/checks.yml` | Syntax, restoration, job-guard, publication checks |
| `.gitignore` / `.gitattributes` | Exclusions and text/binary handling |
| `publication-files.json` | Exact publication file set |
| `patches/ootdiffusion-local.patch` | Pinned-source OpenPose import cleanup |
| `THIRD_PARTY.md` / `LICENSE_POLICY.md` | Provenance, upstream terms, original-code license boundaries |
| `SECURITY.md` | Local-use assumptions and deployment risks |

The [architecture guide](docs/ARCHITECTURE.md) has the expanded internal flow and
runtime contracts. This README contains the full installation procedure.

## HTTP API

This is a local research API, not an authenticated public service.

| Method | Endpoint | Contract |
| --- | --- | --- |
| GET | `/` | Application HTML |
| GET | `/api/model-image` | Configured model photo, or `404` |
| POST | `/upload` | Multipart `file`; returns `filename` and `filepath` |
| POST | `/api/recommend` | JSON `filepath`; properties and ranked candidates |
| POST | `/recommend` | Recommendation alias |
| GET | `/catalog-image/<filename>` | Local original |
| GET | `/catalog-crop/<filename>/<garment_type>` | Local crop, or original fallback |
| POST | `/tryon` | Start job; returns `job_id`, or `409` when busy |
| GET | `/tryon/progress/<job_id>` | Status, progress, message, result URL or error |

**Recommendation request**

```json
{
  "filepath": "upload_example.png"
}
```

**Generation request**

```json
{
  "uploaded_filepath": "upload_example.png",
  "catalog_filename": "a-local-catalog-image.png",
  "uploaded_type": "upper"
}
```

Use `uploaded_type: "lower"` for a pants upload; the catalog item then supplies
the shirt. Successful admission returns `{"job_id": "example-job-id"}`.

**Completed progress response**

```json
{
  "status": "done",
  "progress": 100,
  "message": "Complete!",
  "result_url": "/static/uploads/tryon_example.png"
}
```

IDs and filenames above are illustrative; use actual upload/recommendation
responses. Job states are `running`, `done`, or `error`; they disappear on restart.

## Testing and Verification

### Offline and CI

No data, weights, or GPU are needed:

```powershell
py -3.10 -m venv .test_env
.\.test_env\Scripts\python.exe -m pip install -r requirements-test.txt
.\.test_env\Scripts\python.exe -m compileall -q app.py recommender.py build_index.py garment_cleaner.py scripts
.\.test_env\Scripts\python.exe scripts/test_restore_assets.py
.\.test_env\Scripts\python.exe scripts/test_job_guard.py
.\.test_env\Scripts\python.exe scripts/audit_publication.py --check-only
```

GitHub Actions runs source/offline checks on pushes and pull requests.
A green workflow does **not** certify CUDA inference, image fidelity, or asset rights.

### GPU and Actual API

After installation and supplying permitted fixtures:

```powershell
.\ootd_env\Scripts\python.exe scripts/verify_ootd.py
.\venv\Scripts\python.exe scripts/verify_system.py --skip-tryon
```

Remove `--skip-tryon` for two actual sequential generations. Do not run this mode
alongside browser inference tests or interactive requests.

### Browser Workflows

Keep Flask running, then run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe scripts/test_upload_preview.py --url http://127.0.0.1:5055
.\venv\Scripts\python.exe scripts/capture_screenshots.py --url http://127.0.0.1:5055 --garment upper
```

The capture workflow checks a new generated image and its download response.
Use `--shirt` and `--pants` for alternative permitted fixtures. Raw reports and
data-bearing captures stay ignored, not automatically publication-safe.

Refresh the clean desktop screenshot without data or photos:

```powershell
.\venv\Scripts\python.exe scripts/capture_readme_screenshots.py
```

### Recorded Evidence

Observed on **13 October 2026**, Windows x64, Python 3.13.1 / 3.10.11,
NVIDIA Quadro RTX 4000, 8 GB VRAM:

| Check | Recorded outcome |
| --- | --- |
| Installed environments | `pip check` passed in both |
| GPU environment | 5 checks passed, including real CUDA tensor and inference imports |
| Models | All 30 selected files verified by sizes/checksums |
| Catalog | 10,335 indexed originals and matching masks restored |
| API/catalog workflow | 27 checks passed; duplicate inference skipped |
| Fresh upper-upload outfit | HD + DC completed, output rendered and downloaded |
| Upload-preview timing | 4 real Chrome checks passed |
| Restoration safety | 7 offline tests passed |
| Job admission | 3 isolated tests passed |
| Fresh source bootstrap | Pinned checkout, patch, inference `--help` verified |

A fresh complete lower-upload outfit remains unverified. A subsequent browser
retest was stopped after overlapping with an interactive GPU job; the single-job
guard was added afterward. This is execution evidence, not an exhaustive visual
quality benchmark. See [detailed testing notes](docs/TESTING.md).

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


## Limitations

| Boundary | Current behavior |
| --- | --- |
| Fidelity | Shape, sleeves, color, length, and garment details can change |
| Fit | No size prediction, body measurements, fabric simulation, or fitting guarantee |
| Recommendation accuracy | No calibrated quality metric or compatibility benchmark |
| Broader categories | Detection is broader than the verified shirt/pants generation flow |
| Lower-upload generation | Recommendations checked; fresh full inference still needs verification |
| Properties | Category established; other displayed attributes not reliable |
| Share / views | No completed social sharing or multi-angle generation |
| Job durability | In memory only; no persistence or resume after restart |
| Concurrency | One job per process, not a distributed queue |
| Progress | Approximate, not a reliable time estimate |
| Deployment | No authentication, isolation, cancellation, or rate limiting |
| Coverage | Diversity, fairness, adversarial uploads, extensive fidelity, benchmarks unverified |

An observed output changed recommended shorts into trousers and lengthened shirt
sleeves. Successful execution should not be presented as exact garment preservation.

## Security and Publication

Keep the development server on `127.0.0.1`; public exposure needs a separate
security and deployment review. Image upload and process launching are not
production service isolation.

Environments, upstream runtime source, checkpoints, datasets, derived catalogs,
personal photos, uploads, generated results, logs, archives, and credentials are
excluded. Only reviewed publication images belong in `docs/images/`.

Before publishing a change:

```powershell
.\.test_env\Scripts\python.exe scripts/audit_publication.py --check-only
git diff --check
git diff --stat
```

Review `publication-files.json` and the actual staged diff. Pattern checks reduce
accidental disclosure but do not prove absence of every secret or establish asset
rights. Recommendation/result screenshots need redistribution permission for all
visible and contributing assets; private visibility does not replace permission.

Read [SECURITY.md](SECURITY.md), [publication policy](docs/PUBLISHING.md),
and [third-party terms](THIRD_PARTY.md).

## Contributing

1. Keep changes focused and leave weights, datasets, and personal photos out of commits.
2. Use the appropriate Python environment for the affected module.
3. Run offline tests and publication audit; run GPU/browser checks for affected workflows.
4. Describe changed behavior, performed verification, and remaining limitations.
5. Update this README when setup commands, package pins, or runtime contracts change.

Bug reports should contain reproducible steps, environment versions, and redacted
error summaries. Do not attach credentials, personal images, machine paths, or
restricted catalog material without authorization.

## License and Attribution

Original application code, first-party scripts, and documentation are licensed
under [MIT](LICENSE), copyright 2026 AbdulRehman
([engr-abd-rehman](https://github.com/engr-abd-rehman)).

The OOTDiffusion adaptation remains subject to
[CC BY-NC-SA 4.0](licenses/OOTDiffusion-CC-BY-NC-SA-4.0.txt).
MIT does not relicense upstream code, weights, datasets, or photos and does not
make the whole stack unrestricted for commercial use. See
[license scope](LICENSE_POLICY.md) and [third-party notices](THIRD_PARTY.md).

| Upstream project | Contribution |
| --- | --- |
| [OpenAI CLIP](https://github.com/openai/CLIP) | Pretrained recommendation encoder |
| [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) | Pretrained controllable virtual try-on implementation |
| [OOTDiffusion paper](https://arxiv.org/abs/2403.01779) | Model methodology and research attribution |
| [Text2Human](https://github.com/yumingj/Text2Human) | Processed dataset preparation source |
| [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal#agreement) | Underlying data and provider usage terms |

This project integrates pretrained models; it does not claim to have trained
CLIP or OOTDiffusion. Providers' original terms remain applicable.
