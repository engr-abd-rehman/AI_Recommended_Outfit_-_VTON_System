"""Regression check: wait for FileReader and image loading after each upload."""

import argparse
import base64
import mimetypes
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5055")
    parser.add_argument("--shirt", type=Path, default=ROOT / "static/Products/shirts/maroon-shirt.png")
    parser.add_argument("--pants", type=Path, default=ROOT / "static/Products/pants/beige-pants.png")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        for round_number in range(2):
            page.goto(args.url, wait_until="networkidle")
            for sample in (args.shirt.resolve(), args.pants.resolve()):
                page.locator("#fileInput").set_input_files(str(sample))
                mime = mimetypes.guess_type(sample.name)[0] or "application/octet-stream"
                expected = f"data:{mime};base64," + base64.b64encode(sample.read_bytes()).decode("ascii")
                page.wait_for_function("""expected => {
                    const image = document.getElementById('previewImg');
                    return image.getAttribute('src') === expected && image.complete && image.naturalWidth > 0;
                }""", arg=expected)
                page.locator("#previewImg").evaluate("img => img.decode()")
                assert page.locator("#tryOnBtn").is_disabled()
                print(f"PASS: Upload preview round {round_number + 1}: {sample.name}", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
