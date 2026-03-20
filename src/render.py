"""
render.py — Render HTML/CSS documents to images using a headless browser (Playwright).

Produces deterministic, pixel-perfect renders at a fixed resolution.

Copied from quantitative-text-editing/src/gen_pipeline/render.py and extended with
render_triple() for the three-image stimulus format (source / ground_truth / degraded).
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
from playwright.async_api import async_playwright, Browser


@dataclass
class RenderConfig:
    """Configuration for deterministic rendering."""
    width: int = 1024
    height: int = 1024
    device_scale_factor: float = 1.0  # keep at 1 to avoid DPI scaling artifacts
    # Downscale rendered images to this size. None means no downscaling.
    downscale_to: int | None = 512
    # Disable animations/transitions for determinism
    disable_animations: bool = True
    # Optional: force a specific font to avoid system font differences
    default_font: str | None = None


@dataclass
class RenderResult:
    """Result of rendering an HTML document."""
    image_path: Path
    width: int
    height: int
    html_path: Path | None = None
    errors: list[str] = field(default_factory=list)


class Renderer:
    """Headless browser renderer for HTML→image conversion."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self._playwright = None
        self._browser: Browser | None = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            args=[
                "--disable-gpu",
                "--disable-lcd-text",              # disable subpixel rendering
                "--disable-font-subpixel-positioning",
                "--font-render-hinting=none",      # consistent text rendering
            ]
        )
        return self

    async def __aexit__(self, *exc):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _new_page(self):
        """Create a new page with deterministic settings."""
        context = await self._browser.new_context(
            viewport={"width": self.config.width, "height": self.config.height},
            device_scale_factor=self.config.device_scale_factor,
            reduced_motion="reduce" if self.config.disable_animations else "no-preference",
        )
        page = await context.new_page()

        if self.config.disable_animations:
            await page.add_style_tag(content="""
                *, *::before, *::after {
                    animation-duration: 0s !important;
                    animation-delay: 0s !important;
                    transition-duration: 0s !important;
                    transition-delay: 0s !important;
                }
            """)

        if self.config.default_font:
            await page.add_style_tag(content=f"""
                * {{ font-family: '{self.config.default_font}', sans-serif !important; }}
            """)

        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        return page, context, errors

    def _downscale(self, image_path: Path) -> None:
        """Downscale an image in-place using Lanczos resampling."""
        size = self.config.downscale_to
        if size is None:
            return
        img = Image.open(image_path)
        if img.size == (size, size):
            return
        img = img.resize((size, size), Image.LANCZOS)
        img.save(image_path)

    async def render_html_string(
        self, html: str, output_path: Path, full_page: bool = False
    ) -> RenderResult:
        """Render an HTML string to a PNG image."""
        page, context, errors = await self._new_page()
        try:
            await page.set_content(html, wait_until="networkidle")
            await page.evaluate("() => document.fonts.ready")
            await page.screenshot(path=str(output_path), full_page=full_page, type="png")
            self._downscale(output_path)
            final_size = self.config.downscale_to or self.config.width
            return RenderResult(
                image_path=output_path,
                width=final_size,
                height=final_size,
                errors=errors,
            )
        finally:
            await context.close()

    async def render_triple(
        self,
        source_html: str,
        ground_truth_html: str,
        degraded_html: str,
        output_dir: Path,
        stimulus_id: str,
    ) -> tuple[RenderResult, RenderResult, RenderResult]:
        """Render the three images for one calibration stimulus.

        Returns (source_result, ground_truth_result, degraded_result).
        File names: <stimulus_id>_source.png, <stimulus_id>_ground_truth.png,
                    <stimulus_id>_degraded.png.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        source_result = await self.render_html_string(
            source_html, output_dir / f"{stimulus_id}_source.png"
        )
        gt_result = await self.render_html_string(
            ground_truth_html, output_dir / f"{stimulus_id}_ground_truth.png"
        )
        degraded_result = await self.render_html_string(
            degraded_html, output_dir / f"{stimulus_id}_degraded.png"
        )
        return source_result, gt_result, degraded_result


def render_html_sync(
    html: str,
    output_path: str | Path,
    config: RenderConfig | None = None,
) -> RenderResult:
    """Synchronous wrapper: render a single HTML string to a PNG."""
    async def _run():
        async with Renderer(config) as r:
            return await r.render_html_string(html, Path(output_path))
    return asyncio.run(_run())


def render_triple_sync(
    source_html: str,
    ground_truth_html: str,
    degraded_html: str,
    output_dir: str | Path,
    stimulus_id: str,
    config: RenderConfig | None = None,
) -> tuple[RenderResult, RenderResult, RenderResult]:
    """Synchronous wrapper for render_triple."""
    async def _run():
        async with Renderer(config) as r:
            return await r.render_triple(
                source_html, ground_truth_html, degraded_html,
                Path(output_dir), stimulus_id,
            )
    return asyncio.run(_run())


if __name__ == "__main__":
    # Milestone 0 smoke test — renders a simple hello-world to data/test_render.png.
    import sys
    from pathlib import Path

    out = Path(__file__).parents[1] / "data" / "test_render.png"
    out.parent.mkdir(exist_ok=True)

    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { width: 1024px; height: 1024px; background: #F5F5F5;
       display: flex; justify-content: center; align-items: center; }
h1 { font-family: Arial, sans-serif; font-size: 72px; color: #1A1A1A; }
</style></head><body>
<h1 id="heading">Hello, VLM-Calibration</h1>
</body></html>"""

    result = render_html_sync(html, out)
    print(f"Rendered: {result.image_path}  ({result.width}x{result.height})")
    if result.errors:
        print(f"Console errors: {result.errors}", file=sys.stderr)
        sys.exit(1)
    print("Milestone 0 OK")
