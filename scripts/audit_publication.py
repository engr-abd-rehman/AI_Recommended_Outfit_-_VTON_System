"""Audit an exact source-only file set; never creates or pushes a remote repo."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
SENSITIVE = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("API credential", re.compile(r"\b(?:sk-proj-[A-Za-z0-9_-]{30,}|AIza[0-9A-Za-z_-]{30,}|AKIA[0-9A-Z]{16})")),
    ("machine path", re.compile(r"[A-Za-z]:[\\/](?:Users|Clients)[\\/]")),
)


def git_files():
    if (ROOT / ".git").exists():
        prefix = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    else:
        temporary = ROOT / ".downloads" / "publication-audit.git"
        if not temporary.exists():
            subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "--bare", str(temporary)],
                           check=True, capture_output=True)
        prefix = ["git", f"--git-dir={temporary}", f"--work-tree={ROOT}"]
    result = subprocess.run([*prefix, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            check=True, capture_output=True)
    return {name for name in result.stdout.decode("utf-8").split("\0") if name}


def forbidden(name):
    path = PurePosixPath(name)
    if any(part in {".git", "venv", "ootd_env", ".test_env", ".downloads", ".release", "__pycache__",
                    "Kaggle_datasets", "OOTDiffusion", ".aws", ".codex", ".agents"} for part in path.parts):
        return True
    if name.startswith(("docs/testing/", "docs/screenshots/", "static/uploads/", "static/crops/")):
        return True
    if name.startswith(("static/model/", "static/Products/")) and path.name != ".gitkeep":
        return True
    return path.name in {"catalog_metadata.json", "catalog_vectors.npy"} or path.suffix.lower() in {
        ".log", ".part", ".zip", ".pem", ".key", ".p12", ".pfx", ".pyc", ".npy"} or path.name.startswith(".env")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stage", action="store_true", help="Copy only audited files into the ignored release folder")
    parser.add_argument("--private-term", action="append", default=[], help="Additional local-only privacy checks")
    args = parser.parse_args()
    publication = json.loads((ROOT / "publication-files.json").read_text(encoding="utf-8"))
    names = publication["files"]
    expected = set(names)
    issues = []
    if len(names) != len(expected):
        issues.append("Duplicate allowlist entries")
    candidates = git_files()
    for name in sorted(candidates - expected):
        issues.append(f"Unexpected non-ignored file: {name}")
    manifest = []
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or forbidden(name):
            issues.append(f"Forbidden allowlist entry: {name}")
            continue
        path = ROOT.joinpath(*relative.parts)
        if not path.is_file() or ROOT not in path.resolve().parents or path.is_symlink():
            issues.append(f"Missing or unsafe file: {name}")
            continue
        if name not in candidates:
            issues.append(f"Required file is ignored: {name}")
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            issues.append(f"Oversized file needs review: {name}")
        manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".html", ".css", ".js", ".yml", ".patch"}:
            content = data.decode("utf-8", errors="replace")
            for category, pattern in SENSITIVE:
                for match in pattern.finditer(content):
                    line = content.count("\n", 0, match.start()) + 1
                    issues.append(f"Potential {category}: {name}:{line} (content redacted)")
            for term in args.private_term:
                if term and term.casefold() in content.casefold():
                    issues.append(f"Private term found: {name} (term redacted)")
            if path.suffix.lower() == ".md":
                for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                    target = target.strip().strip("<>").split("#", 1)[0]
                    if not target or "://" in target:
                        continue
                    linked = (path.parent / unquote(target)).resolve()
                    if ROOT not in linked.parents or not linked.is_file():
                        issues.append(f"Broken or unsafe local link: {name}: {target}")
                    elif linked.relative_to(ROOT).as_posix() not in expected:
                        issues.append(f"Link target is not published: {name}: {target}")
    report = {"status": "FAIL" if issues else "PASS", "files": manifest,
              "file_count": len(manifest), "total_bytes": sum(item["bytes"] for item in manifest), "issues": issues}
    if not args.check_only:
        output = ROOT / ".downloads" / "publication-manifest.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for issue in issues:
        print(f"FAIL: {issue}")
    if issues:
        return 1
    if args.stage:
        repo_name = publication["repository"].rsplit("/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name) or repo_name in {".", ".."}:
            raise RuntimeError("Unsafe release repository name")
        stage = ROOT / ".release" / repo_name
        stage.mkdir(parents=True, exist_ok=True)
        if ROOT not in stage.resolve().parents:
            raise RuntimeError("Release folder resolves outside this project")
        extra = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
                 and ".git" not in path.relative_to(stage).parts} - expected
        if extra:
            raise RuntimeError("Release folder contains unexpected files; not deleting or publishing them")
        for name in names:
            target = stage / name
            if stage.resolve() not in target.resolve().parents:
                raise RuntimeError("Release destination escapes its folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        print(f"Staged exact release: .release/{repo_name}")
    print(f"PASS: {len(manifest)} files, {report['total_bytes'] / 1e6:.2f} MB; no remote actions performed")
    print("Pattern checks reduce risk but do not certify absence of every secret or establish asset rights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
