"""Fetch the pinned upstream source and apply the documented local patch."""

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/levihsu/OOTDiffusion.git"
REVISION = "13ef0faba266cdde9febc8ad39be2395bbb89d9c"


def git(directory, *arguments, capture=False):
    return subprocess.run(["git", "-c", f"safe.directory={directory.as_posix()}", "-C", str(directory), *arguments],
                          check=True, text=True, capture_output=capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=ROOT / "OOTDiffusion")
    args = parser.parse_args()
    target = args.target.resolve()
    if ROOT not in target.parents:
        parser.error("Target must be a child directory inside this project")
    if target.exists():
        revision = git(target, "rev-parse", "HEAD", capture=True).stdout.strip()
        if revision != REVISION:
            raise SystemExit("Existing source is on another revision; not overwriting it. Use a new --target directory.")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(target)], check=True)
        git(target, "checkout", "--detach", REVISION)
    patch = ROOT / "patches" / "ootdiffusion-local.patch"
    reverse = subprocess.run(["git", "-c", f"safe.directory={target.as_posix()}", "-C", str(target),
                              "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if reverse.returncode != 0:
        git(target, "apply", "--check", str(patch))
        git(target, "apply", str(patch))
    (target / "run" / "images_output").mkdir(exist_ok=True)
    print(f"OOTDiffusion ready at {target}; pinned revision {REVISION}")


if __name__ == "__main__":
    main()
