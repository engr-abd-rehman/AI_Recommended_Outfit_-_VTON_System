"""Capture real browser workflows, waiting for each generation to finish."""

import argparse
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5055")
    parser.add_argument("--tryon-timeout", type=int, default=1800)
    parser.add_argument("--shirt", type=Path, default=ROOT / "static/Products/shirts/maroon-shirt.png")
    parser.add_argument("--pants", type=Path, default=ROOT / "static/Products/pants/beige-pants.png")
    parser.add_argument("--garment", choices=("both", "upper", "lower"), default="both")
    args = parser.parse_args()
    screenshots = ROOT / "docs" / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    checks = []
    errors = []
    output = ROOT / "docs" / "testing" / "browser-checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    def write_report(phase):
        failures = sum(item["status"] == "FAIL" for item in checks)
        output.write_text(json.dumps({"url": args.url, "started_at": started,
                                     "phase": phase, "checks": checks,
                                     "passed": len(checks) - failures, "failed": failures}, indent=2), encoding="utf-8")

    write_report("running")

    def record(name, passed, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        write_report("running")
        print(f"{checks[-1]['status']}: {name}: {detail}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(10000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.screenshot(path=str(screenshots / "01-home-desktop.png"), full_page=True)
        record("Homepage title", "Virtual Try-On" in page.title(), page.title())
        record("Stylesheet hidden states", not page.locator("#tryonLoadingOverlay").is_visible(),
               "Loading overlay is hidden before a try-on request")
        page.locator("#getRecoBtn").click()
        record("Missing upload warning", page.get_by_text("Please upload a clothing image first!", exact=True).is_visible(),
               "Warning appears when no image is selected")

        # Use a real browser DataTransfer to exercise the advertised drop workflow.
        sample = args.shirt.resolve()
        mime = mimetypes.guess_type(sample.name)[0] or "application/octet-stream"
        import base64
        transfer = page.evaluate_handle("""({data, name, mime}) => {
            const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], name, {type: mime}));
            return transfer;
        }""", {"data": base64.b64encode(sample.read_bytes()).decode("ascii"), "name": sample.name, "mime": mime})
        page.locator("#uploadBox").dispatch_event("drop", {"dataTransfer": transfer})
        record("Drag-and-drop file selection", page.locator("#fileInput").evaluate("el => el.files.length") == 1,
               "Dropped file must be stored in the file input for submission")

        for sample_path, label in ((args.shirt.resolve(), "upper"), (args.pants.resolve(), "lower")):
            if args.garment not in ("both", label):
                continue
            page.locator("#fileInput").set_input_files(str(sample_path))
            record(f"New upload clears try-on selection {label}", page.locator("#tryOnBtn").is_disabled(),
                   "Previous recommendations cannot trigger try-on for a new upload")
            mime = mimetypes.guess_type(sample_path.name)[0] or "application/octet-stream"
            expected_preview = f"data:{mime};base64," + base64.b64encode(sample_path.read_bytes()).decode("ascii")
            page.wait_for_function("""expected => {
                const image = document.getElementById('previewImg');
                return image.getAttribute('src') === expected && image.complete && image.naturalWidth > 0;
            }""", arg=expected_preview)
            page.screenshot(path=str(screenshots / f"02-upload-{label}.png"), full_page=True)
            with page.expect_response(lambda response: response.url.endswith("/api/recommend"), timeout=60000) as pending:
                page.locator("#getRecoBtn").click()
            response = pending.value
            result = response.json()
            page.wait_for_function("!document.getElementById('getRecoBtn').disabled", timeout=60000)
            page.wait_for_function("Array.from(document.querySelectorAll('.slot-img')).every(img => !img.getAttribute('src') || img.complete)")
            page.wait_for_timeout(500)
            if result.get("success"):
                page.get_by_text("compatible outfits found!", exact=False).wait_for(state="hidden", timeout=5000)
            page.screenshot(path=str(screenshots / f"03-recommendations-{label}.png"), full_page=True)
            record(f"Browser recommendations {label}", result.get("success") and len(result.get("recommendations", [])) == 4,
                   {"http_status": response.status, "query_type": result.get("query_type"), "error": result.get("error")})
            if result.get("success"):
                image_ids = [f"#{'bottom' if label == 'upper' else 'top'}Img{i}" for i in range(4)]
                loaded = [page.locator(selector).evaluate("img => img.complete && img.naturalWidth > 0") for selector in image_ids]
                record(f"Visible recommendation images {label}", all(loaded), loaded)
                with page.expect_response(lambda response: response.url.endswith("/tryon"), timeout=10000) as pending_tryon:
                    if label == "upper":
                        page.locator("#outfitCard1").click()
                        page.locator("#tryOnBtn").click()
                    else:
                        page.locator("#outfitCard0 .btn-select").click()
                tryon_response = pending_tryon.value
                selected = 1 if label == "upper" else 0
                record(f"Try-on control uses selected recommendation {label}",
                       tryon_response.request.post_data_json.get("catalog_filename") == result["recommendations"][selected]["filename"],
                       tryon_response.request.post_data_json)
                job = tryon_response.json()
                record(f"Try-on request {label}", bool(job.get("job_id")), job)
                if job.get("job_id"):
                    job_id = job["job_id"]
                    deadline = time.monotonic() + args.tryon_timeout
                    page.screenshot(path=str(screenshots / f"04-generating-{label}.png"), full_page=True)
                    last_message = None
                    while True:
                        job = context.request.get(f"{args.url}/tryon/progress/{job_id}").json()
                        message = job.get("message")
                        if message != last_message:
                            print(f"Generation {label}: {job.get('progress')}%: {message}", flush=True)
                            last_message = message
                        if job.get("status") != "running":
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Generation timed out; stopping before starting another job")
                        page.wait_for_timeout(2000)
                    record(f"Completed generation {label}", job.get("status") == "done", job)
                    page.wait_for_function("!document.getElementById('tryOnBtn').disabled", timeout=10000)
                    if job.get("status") == "done":
                        page.locator("#tryonImg").evaluate("img => img.decode()")
                        actual = page.locator("#tryonImg").get_attribute("src")
                        record(f"New result rendered {label}", actual.startswith(job["result_url"]), actual)
                        image_response = context.request.get(args.url + job["result_url"])
                        record(f"Result image downloadable {label}", image_response.ok
                               and image_response.headers.get("content-type", "").startswith("image/"),
                               {"http_status": image_response.status, "bytes": len(image_response.body())})
                        (screenshots / f"06-generated-outfit-{label}.png").write_bytes(image_response.body())
                        page.wait_for_timeout(3500)
                page.screenshot(path=str(screenshots / f"04-tryon-{label}.png"), full_page=True)
            page.reload(wait_until="networkidle")

        page.set_viewport_size({"width": 390, "height": 844})
        page.reload(wait_until="networkidle")
        page.screenshot(path=str(screenshots / "05-home-mobile.png"), full_page=True)
        layout = page.evaluate("""() => ({
            width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
            upload: document.getElementById('uploadBox').getBoundingClientRect().toJSON(),
            tryon: document.getElementById('tryonImg').getBoundingClientRect().toJSON()
        })""")
        record("Mobile layout fits viewport", layout["scrollWidth"] <= layout["width"]
               and layout["upload"]["right"] <= layout["width"] and layout["upload"]["width"] >= 250
               and layout["tryon"]["right"] <= layout["width"] and layout["tryon"]["width"] >= 250, layout)
        record("No browser JavaScript errors", not errors, errors)
        browser.close()

    failures = sum(item["status"] == "FAIL" for item in checks)
    write_report("complete")
    print(f"Screenshots: {screenshots}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        output = ROOT / "docs" / "testing" / "browser-checks.json"
        if output.is_file():
            report = json.loads(output.read_text(encoding="utf-8"))
            report["checks"].append({"name": "Browser automation completed", "status": "FAIL", "detail": str(error)})
            report["failed"] += 1
            report["phase"] = "aborted"
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
