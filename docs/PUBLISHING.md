# Publication Checklist

Repository: **`engr-abd-rehman/AI_Recommended_Outfit_-_VTON_System`**.
The owner confirmed publishing authority, requested the push, and approved MIT
for original code. The existing repository is Private; its visibility is preserved.
Future releases still require the appropriate owner's authorization.

## Exact Payload

[publication-files.json](../publication-files.json) is the explicit allowlist.
It includes first-party application source, dependency declarations, setup and
architecture/testing/security notes, the upstream-derived patch and its license,
offline tests/CI, empty local-asset placeholders, and two publication-safe UI
screenshots. Files not on the list are not copied to the release payload.

The full downloaded upstream tree is not pushed as a nested Git repository.
Setup fetches its pinned revision and applies the recorded patch. Nothing is
deleted from the original working project to make the release smaller.

## Deliberate Exclusions

- Both environments, caches, compiled files, installers, and download archives.
- Dataset images, masks, annotations, derived metadata, and vector index.
- Diffusion, parsing, pose, and CLIP weights.
- Stock/model photos, product fixtures, personal uploads, crops, generated outputs.
- Raw reports, machine paths, logs, and data-bearing screenshots.
- Tool configuration, credential files, `.env` files, and private keys.
- Old handoff notes and upstream example assets.

The downloaded assets and local photos remain available on the original machine.
Fresh installations restore them independently using the documented sources.
Do not override exclusions with `git add -f` or push a whole workspace archive.

## Audit and Stage

```powershell
.\venv\Scripts\python.exe scripts/audit_publication.py --stage
```

The audit checks unexpected non-ignored files, required files, size thresholds,
suspicious credentials/machine paths, and local documentation links. It emits
an exact byte-count/SHA-256 manifest under ignored `.downloads/` and stages only
approved files in `.release/AI_Recommended_Outfit_-_VTON_System/`. It does not create a
remote, authenticate to GitHub, commit, or push. Secret-pattern checks are not a
guarantee; inspect the file list and asset provenance as well.

## Final Approval Before Push

Confirm all of these before any upload:

1. Correct GitHub owner and repository name.
2. Public or private visibility.
3. Authority to publish the first-party code, including any client obligations.
4. License for original code; upstream-derived files remain under upstream terms.
5. Acceptance of the exact audited file manifest and source-only screenshots.

Prefer publishing from the audited release directory. For an existing local
clone that contains ignored runtime assets, stage only explicit allowlist paths,
verify the entire commit tree and reachable history, and review sizes before
pushing `main`. Never stage the whole folder blindly, force-push an existing
repository, or replace its history. This release retains the original initial commit.

If the connected GitHub tooling cannot create a new repository, the owner can
create an empty repository with the approved name/visibility and provide its URL.
Do not paste access tokens into chat or add them to the remote URL.

## After Publishing

Verify the remote file tree against the manifest, README image rendering, links,
upstream attribution, setup commands, and CI results. Test a clean checkout with
the documented source bootstrap. Hosted CI proves only its executed offline
checks, not GPU inference or production readiness.
