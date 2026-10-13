# Architecture and File Guide

## Runtime Boundaries

| Environment | Python | Responsibility | Key tested versions |
| --- | --- | --- | --- |
| `venv` | 3.13 | Flask, CLIP recommendations, index building, browser tools | Torch 2.11.0 CPU, TorchVision 0.26.0 CPU, NumPy 2.4.3 |
| `ootd_env` | 3.10 | GPU pose/parsing and diffusion; asset restoration | Torch 2.0.1 CUDA 11.8, TorchVision 0.15.2, NumPy 1.26.4, Diffusers 0.24.0 |

Flask launches the inference interpreter explicitly. No shell activation is
passed between processes. This separation avoids mixing incompatible stacks.
CLIP here is a locally loaded model, not an external API request.

## First-Party Files

| File | What It Does |
| --- | --- |
| `app.py` | Serves Flask/Jinja UI, validates uploads, serves photos/crops, dispatches generation, stores job progress, admits one running generation per process |
| `recommender.py` | Loads CLIP and the local index lazily; determines garment type, filters complementary categories, ranks and deduplicates candidates |
| `build_index.py` | Scans training images in sorted order, reads annotations, encodes batches, L2-normalizes vectors, writes both runtime catalog files |
| `garment_cleaner.py` | Scores upper-garment symmetry/occlusion/size for ranking; also exposes optional skin-removal/inpainting helpers |
| `templates/index.html` | Main application controls and layout; generic local-demo identity, not an authentication implementation |
| `static/js/main.js` | Validates selection, handles file/drop input, calls upload/recommendation endpoints, tracks selected card, starts jobs, polls results, downloads images |
| `static/css/style.css` | Layout, loading/result states, garment card styles, and responsive breakpoints |
| `requirements.txt` | Main application dependency versions |
| `ootd_requirements.txt` | Separate GPU/inference dependency versions |
| `requirements-assets.txt` | Small download-tool dependency set |
| `requirements-dev.txt` | Playwright for local Chrome checks |
| `requirements-test.txt` | Lightweight Python 3.10 offline/CI test dependencies; no Torch or models |
| `.github/workflows/checks.yml` | Offline source, restoration, job-admission, and publication checks |
| `.gitignore` | Excludes local assets, personal photos, runtime reports, credentials, and environments |
| `.gitattributes` | Normalizes text line endings; marks images/vector files as binary |
| `publication-files.json` | Explicit file allowlist for publication |

## Scripts

| Script | Purpose |
| --- | --- |
| `setup_ootdiffusion.py` | Downloads pinned official source, verifies the checked-out revision, applies the small local patch, creates the output folder |
| `restore_assets.py` | Downloads pinned model files, resumes HTTP transfers, verifies hashes/sizes, fetches the processed dataset, safely extracts matching data |
| `restore_and_test.py` | Coordinates source/assets, builds a missing index, runs environment/API/browser checks, generates local reports |
| `verify_ootd.py` | Checks NumPy/Torch integration, CUDA, a real GPU tensor operation, pip consistency, and inference imports; does not load diffusion weights |
| `verify_system.py` | Exercises real Flask routes, upload validation, index alignment, catalog originals, model readiness, recommendations, image endpoints, and optional actual generation |
| `capture_screenshots.py` | Runs real Chrome workflows and sequential generation, verifies result rendering/download, captures local data-bearing screenshots |
| `test_upload_preview.py` | Regression test for asynchronous FileReader/image loading after changing uploads |
| `test_restore_assets.py` | Offline archive safety, fresh-clone extraction, hashing, missing-data rejection, and resumable-transfer tests |
| `test_job_guard.py` | Isolated route-admission tests with mock workers; does not imply GPU inference was tested |
| `capture_readme_screenshots.py` | Starts an isolated source-only preview and captures publication-safe desktop/mobile UI without photos or catalog data |
| `audit_publication.py` | Checks the allowlist, Git exclusions, file sizes, links, and suspicious content; emits hashes and optionally stages an exact release copy |

## Downloaded and Generated Files

These exist locally after setup, but are deliberately not repository payload:

| Location | Producer or Source | Consumer |
| --- | --- | --- |
| `OOTDiffusion/` | Pinned upstream Git checkout + local patch | Inference subprocess |
| `OOTDiffusion/checkpoints/` | Official OOTDiffusion/CLIP repositories | HD/DC, VAE, text/image encoders, parsing, OpenPose |
| `Kaggle_datasets/train_images/` | Processed Text2Human archive | Index builder, recommendation crops, inference garment preparation |
| `Kaggle_datasets/segm/` | Same archive | Garment-region previews, hi-res extraction, quality scoring |
| `Kaggle_datasets/shape_ann/` and `texture_ann/` | Same archive | Locally derived catalog metadata |
| `catalog_vectors.npy` | `build_index.py` | Recommendation search; tested shape `(10335, 512)` |
| `catalog_metadata.json` | `build_index.py` | Categories, filenames, garment types, annotation lookup |
| `static/model/base_model.jpg` or `.png` | Operator-supplied permitted photo | UI and first inference pass |
| `static/Products/` | Operator-supplied permitted test images | Repeatable workflow checks |
| `static/uploads/` | Upload/generation handlers | Uploaded files, normalized garments, intermediate/final results |
| `static/crops/` | Crop endpoint | Cached transparent garment previews |
| `OOTDiffusion/run/images_output/` | Diffusion subprocess | Mask and `out_hd_0.png`/`out_dc_0.png` |
| `.downloads/` | Restoration/audit helpers | Archive, manifests, private publication reports and smoke-test copies |
| `docs/testing/`, `docs/screenshots/` | Local tests | Raw machine-specific reports and data-bearing screenshots |
| `.release/` | Publication audit | Exact audited source-only release copy |

## Recommendation Flow

1. Verify image content before saving a UUID-named upload.
2. Determine category using metadata, filename hints, image-region heuristics,
   then CLIP text prompts as fallbacks. UUID uploads do not preserve original
   filename hints, so image-based detection is important.
3. Encode the image into a normalized CLIP embedding.
4. Restrict compatible categories: upper to lower, lower to upper, outer/full
   to upper/lower. Retrieve using NumPy vector similarity, not an external search service.
5. Apply front-view and garment-quality heuristics, then deduplicate candidates.
6. Return four recommendations and local crop URLs. Scores are presentation
   heuristics, not a measured compatibility benchmark.

## Generation Flow

The tested two-pass workflow handles a shirt and pants. A lower upload swaps
which garment comes from the user and which comes from the catalog.

1. Resolve inputs and atomically admit one active job in the Flask process.
2. Prepare RGB JPEGs; extract catalog garments using original-resolution masks
   and place them on a 768 x 1024 white canvas where possible.
3. Run upstream HD category 0 against the supplied model photo.
4. Copy its result to the intermediate image.
5. Run DC category 1 against that intermediate result.
6. Copy the DC result directly to a unique result PNG and update job status.

The historical compositor helper is not used in the current final-output path.
The second pass can still change the first garment. The UI's broader category
detection does not establish correct dress/outerwear generation support.

## HTTP Endpoints

| Method | Path | Contract |
| --- | --- | --- |
| GET | `/` | HTML application |
| GET | `/api/model-image` | Local model photo, or 404 when absent |
| POST | `/upload` | Multipart field `file`; returns `filename` and server `filepath` |
| POST | `/api/recommend` or `/recommend` | JSON `filepath`; accepts a returned upload filename or path |
| GET | `/catalog-image/<filename>` | Locally downloaded original |
| GET | `/catalog-crop/<filename>/<garment_type>` | Cached/generated crop; original fallback |
| POST | `/tryon` | `uploaded_filepath`, `catalog_filename`, `uploaded_type`; returns `job_id` or 409 when busy |
| GET | `/tryon/progress/<job_id>` | `status`, `progress`, `message`; on completion `result_url`; on failure `error` |

```json
{
  "uploaded_filepath": "upload_example.png",
  "catalog_filename": "a-local-catalog-image.png",
  "uploaded_type": "upper"
}
```

Those filenames are illustrative, not shipped assets. Use actual responses
from upload/recommendation endpoints. Job state is in memory: restarting Flask
loses it. The lock and shared output names are per-process, not a distributed
queue or multi-worker deployment design. Progress parsing also sees upstream
loading bars, so intermediate percentages are approximate.
