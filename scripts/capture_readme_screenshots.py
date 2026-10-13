"""Capture the real interface in a clean source-only preview, without photos."""

from pathlib import Path
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    preview = ROOT / ".downloads" / "docs-preview"
    images = ROOT / "docs" / "images"
    preview.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "templates/index.html", "static/css/style.css", "static/js/main.js"):
        target = preview / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    if any((preview / "static/model").glob("*.jpg")) or any((preview / "static/model").glob("*.png")):
        raise RuntimeError("Preview unexpectedly contains a model photo; refusing to capture it")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    python = ROOT / "venv" / "Scripts" / "python.exe"
    with (preview / "server.log").open("wb") as log:
        process = subprocess.Popen([str(python), "-m", "flask", "--app", "app", "run", "--host", "127.0.0.1",
                                    "--port", str(port), "--no-reload"], cwd=preview, stdout=log, stderr=log)
        try:
            for attempt in range(30):
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None:
                        raise RuntimeError("Documentation preview server exited")
                    time.sleep(1)
            else:
                raise RuntimeError("Documentation preview did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
                page = browser.new_page(viewport={"width": 1600, "height": 1050})
                page.goto(url, wait_until="networkidle")
                page.locator("#modelFallback").wait_for(state="visible")
                assert page.locator(".user-name").inner_text() == "Local Demo"
                assert not page.locator("#tryonLoadingOverlay").is_visible()
                assert page.locator("#tryonImg").evaluate("img => img.naturalWidth === 0")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                image_path = images / "interface-desktop.png"
                page.screenshot(path=str(image_path), full_page=True)
                print(f"PASS: privacy-safe desktop capture: {image_path.relative_to(ROOT)}", flush=True)
                browser.close()
        finally:
            if process.poll() is None and os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            elif process.poll() is None:
                process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
