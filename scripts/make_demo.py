"""Capture the interactive preview as README screenshots, a GIF, and an MP4.

This script is a build helper, not part of the benchmark. It drives the
generated ``preview.html`` with a headless browser and converts the recorded
tour with ffmpeg. Every visual is therefore a capture of the real page, not a
mock-up.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview.html"
ASSETS = ROOT / "docs" / "assets"
VIDEO_SIZE = {"width": 1280, "height": 720}


def _scroll(page, selector: str, wait: int = 900) -> None:
    page.eval_on_selector(selector, "el => el.scrollIntoView({behavior: 'smooth', block: 'start'})")
    page.wait_for_timeout(wait)


def capture_stills() -> list[Path]:
    from playwright.sync_api import sync_playwright

    ASSETS.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(PREVIEW.as_uri())
        page.wait_for_timeout(1400)
        shots = [
            ("preview-hero.png", None),
            ("preview-results.png", "#results"),
            ("preview-learn.png", "#learn"),
            ("preview-business.png", "#business"),
        ]
        for name, selector in shots:
            if selector:
                _scroll(page, selector)
            path = ASSETS / name
            page.screenshot(path=str(path))
            outputs.append(path)
        _scroll(page, "#quiz")
        page.locator(".option").first.click()
        page.locator("#quizCheck").click()
        page.wait_for_timeout(500)
        path = ASSETS / "preview-quiz.png"
        page.screenshot(path=str(path))
        outputs.append(path)

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(PREVIEW.as_uri())
        mobile.wait_for_timeout(1200)
        path = ASSETS / "preview-mobile.png"
        mobile.screenshot(path=str(path))
        outputs.append(path)
        browser.close()
    return outputs


def capture_video() -> Path:
    from playwright.sync_api import sync_playwright

    tmp = Path(tempfile.mkdtemp(prefix="fbench-demo-"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=VIDEO_SIZE,
            record_video_dir=tmp,
            record_video_size=VIDEO_SIZE,
        )
        page = context.new_page()
        page.goto(PREVIEW.as_uri())
        page.wait_for_timeout(1800)
        page.mouse.move(1050, 380)
        page.wait_for_timeout(900)
        _scroll(page, "#results", 1200)
        page.wait_for_timeout(1900)
        _scroll(page, "#evidence", 1000)
        page.locator(".tab", has_text="E4").click()
        page.wait_for_timeout(1900)
        _scroll(page, "#learn", 1000)
        page.locator("#learnSteps summary").first.click()
        page.wait_for_timeout(500)
        page.locator("#learnSteps input[type=checkbox]").first.check()
        page.wait_for_timeout(1200)
        _scroll(page, "#quiz", 1000)
        page.locator(".option").first.click()
        page.locator("#quizCheck").click()
        page.wait_for_timeout(2100)
        page.locator("#quizSkip").click()
        page.wait_for_timeout(1100)
        _scroll(page, "#business", 1000)
        page.wait_for_timeout(1900)
        context.close()
        browser.close()
    videos = sorted(tmp.glob("*.webm"))
    if not videos:
        raise RuntimeError("browser recorded no video")
    return videos[-1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_gif(video: Path, max_bytes: int = 4_500_000) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for docs/assets/demo.gif")
    settings = [(760, 9, 128), (680, 8, 96), (600, 7, 64)]
    target = ASSETS / "demo.gif"
    for width, fps, colors in settings:
        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "0.6",
                "-i",
                str(video),
                "-filter_complex",
                f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
                f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
                f"[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                "-loop",
                "0",
                str(target),
            ]
        )
        if target.stat().st_size <= max_bytes:
            break
    return target


def make_mp4(video: Path) -> Path:
    target = ASSETS / "demo.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0.6",
            "-i",
            str(video),
            "-vf",
            "fps=30,scale=1280:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-video", action="store_true", help="only refresh the still images")
    args = parser.parse_args()
    if not PREVIEW.exists():
        raise SystemExit("preview.html is missing; run scripts/build_preview.py first")
    for path in capture_stills():
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    if args.skip_video:
        return
    video = capture_video()
    try:
        gif = make_gif(video)
        mp4 = make_mp4(video)
        print(f"wrote {gif.relative_to(ROOT)} ({gif.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"wrote {mp4.relative_to(ROOT)} ({mp4.stat().st_size / 1024 / 1024:.2f} MB)")
    finally:
        shutil.rmtree(video.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
