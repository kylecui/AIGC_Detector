"""Capture real UI screenshots for software copyright 文档鉴别材料.

Requires: server running at http://127.0.0.1:8000
Usage: uv run python scripts/capture_copyright_screenshots.py

Output: docs/software-copyright/screenshots/*.png
"""
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path("docs/software-copyright/screenshots")
SAMPLE_TXT = Path("tmp/copyright_screenshot_sample.txt")

ZH_AI_TEXT = (
    "人工智能技术的快速发展正在深刻改变现代社会的信息传播格局。基于大语言模型的内容生成能力"
    "已经达到了前所未有的水平，这同时也给信息真实性的验证带来了严峻挑战。在这样的背景下，构建"
    "高效可靠的AI生成内容检测系统显得尤为重要。本文从统计特征分析、语言学文体建模以及深度语义"
    "理解三个维度出发，系统阐述了多模型集成检测框架的设计思路与实现路径。实验结果表明，该框架在"
    "多种语言环境和文体类型上均取得了良好的检测效果，为维护网络信息生态安全提供了有力的技术支撑。"
)


async def wait_detect_done(page, timeout_ms: int = 300_000):
    """Wait until Copy Result JSON button becomes enabled (result ready, not loading)."""
    await page.locator("button", has_text="Copy Result JSON").wait_for(
        state="visible", timeout=timeout_ms
    )
    await page.wait_for_function(
        """() => {
            const btns = [...document.querySelectorAll('button')];
            const b = btns.find(x => x.textContent.includes('Copy Result JSON'));
            return b && !b.disabled;
        }""",
        timeout=timeout_ms,
    )
    await page.wait_for_timeout(1200)  # let segment views finish rendering


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLE_TXT.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_TXT.write_text(ZH_AI_TEXT, encoding="utf-8")

    shots = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        page = await browser.new_page(
            viewport={"width": 1440, "height": 900}, locale="zh-CN"
        )

        # -- 01 WebUI home (text mode, empty state) --------------------
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.screenshot(path=OUT / "01-webui-home.png", full_page=True)
        shots.append("01-webui-home.png")

        # -- 02 health check -------------------------------------------
        await page.get_by_role("button", name="Check Service Health").click()
        await page.locator(".status-row .tag").first.wait_for(timeout=30_000)
        await page.wait_for_timeout(500)
        await page.screenshot(path=OUT / "02-health-check.png", full_page=True)
        shots.append("02-health-check.png")

        # -- 03 zh AI-style example loaded (segments on) ----------------
        await page.locator(".example-btn", has_text="中文 AI 风格示例").click()
        cb = page.locator("input[type='checkbox']")
        if not await cb.is_checked():
            await cb.check()
        await page.wait_for_timeout(400)
        await page.screenshot(path=OUT / "03-zh-ai-input.png", full_page=True)
        shots.append("03-zh-ai-input.png")

        # -- 04 zh detection result (with segment analysis) -------------
        await page.get_by_role("button", name="Detect", exact=True).click()
        await wait_detect_done(page)
        await page.screenshot(path=OUT / "04-zh-ai-result.png", full_page=True)
        shots.append("04-zh-ai-result.png")

        # -- 05 en AI-style example result -------------------------------
        await page.get_by_role("button", name="Clear").click()
        await page.wait_for_timeout(400)
        await page.locator(".example-btn", has_text="English AI-style Example").click()
        await page.wait_for_timeout(400)
        await page.get_by_role("button", name="Detect", exact=True).click()
        await wait_detect_done(page)
        await page.screenshot(path=OUT / "05-en-ai-result.png", full_page=True)
        shots.append("05-en-ai-result.png")

        # -- 06 file upload mode (TXT selected, extracted) ---------------
        await page.get_by_role("button", name="Clear").click()
        await page.wait_for_timeout(400)
        await page.locator(".mode-tab", has_text="上传文件").click()
        await page.locator("input.file-input-hidden").set_input_files(
            str(SAMPLE_TXT.resolve())
        )
        await page.wait_for_timeout(1500)  # allow text extraction preview
        await page.screenshot(path=OUT / "06-file-upload-selected.png", full_page=True)
        shots.append("06-file-upload-selected.png")

        # -- 07 file detection result ------------------------------------
        await page.get_by_role("button", name="Detect", exact=True).click()
        await wait_detect_done(page)
        await page.screenshot(path=OUT / "07-file-detect-result.png", full_page=True)
        shots.append("07-file-detect-result.png")

        # -- 08 Swagger UI ------------------------------------------------
        await page.goto(f"{BASE}/docs", wait_until="load")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=OUT / "08-swagger-docs.png", full_page=True)
        shots.append("08-swagger-docs.png")

        # -- 09 ReDoc -----------------------------------------------------
        await page.goto(f"{BASE}/redoc", wait_until="load")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=OUT / "09-redoc.png", full_page=True)
        shots.append("09-redoc.png")

        await browser.close()

    print("CAPTURED:")
    for name in shots:
        f = OUT / name
        size = f.stat().st_size if f.exists() else 0
        print(f"  {name}  {size/1024:.0f} KB")
    missing = [s for s in shots if not (OUT / s).exists() or (OUT / s).stat().st_size < 10_000]
    if missing:
        print(f"SUSPICIOUS (too small / missing): {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(__import__("asyncio").run(main()))
