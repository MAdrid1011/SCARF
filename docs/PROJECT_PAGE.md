# SCARF project page

The static project page is `docs/index.html`. It uses `site.css`, `site.js`, and
the self-contained assets in `docs/media/`. The page can be previewed locally
with `python3 -m http.server 8765 --directory docs`.

The published URL is `https://madrid1011.github.io/SCARF/`. GitHub Pages uses
**GitHub Actions** as its source. `.github/workflows/pages.yml` validates page
changes in pull requests and deploys changes merged into `main`. It can also
be run manually from the Actions tab.

The deployment package contains only `index.html` and local assets referenced
by the page and its CSS. Other technical documents, figure sources, and asset
generation scripts remain in the repository. Validate and package locally:

```bash
node --check docs/site.js
python3 docs/scripts/build_project_page.py --output _site
```

Use an empty output directory. Keep relative asset paths so the page works
under `/SCARF/`. The canonical URL and sharing metadata in `index.html`, the
README links, and the repository's About website should use the published URL.

The presentation links in the hero and talk section point to the published
Xiaohongshu video. Its opening frame is stored locally as `media/talk-cover.jpg`
so the page does not depend on a third-party image endpoint.

The bundled paper PDF and figures came from the local MICRO 2026 camera-ready
sources. Update those assets together when the official paper changes. The
Artifact links use the evaluated SCARF-AE v1.0.4 record,
`https://doi.org/10.5281/zenodo.22434931`.

The method section includes the complete dataflow and architecture figures,
with editable FSDR and SAES schematics in `docs/figures/`. Each schematic has
a separate mobile layout; `picture` selects it below 700px. The BibTeX block
and template attribution follow the VGGT page's presentation.

Maintainers can regenerate assets from the repository root in this order.
The PDF exporter requires the original `micro59-camera-ready/figures/`
directory, which is not part of this public checkout. Normal deployment uses
the committed exports and does not require the paper sources or draw.io.

```bash
uv run --no-project --with pymupdf==1.26.3 --with pillow python docs/scripts/prepare_page_assets.py
python3 docs/scripts/build_mechanism_diagrams.py --export
```

The first script reads the camera-ready PDFs without modifying them, extracts
the unframed scene image, and exports full diagrams and qualitative panels.
It expands the qualitative PDF's in-memory bounds to recover clipped axis
labels and uses the original color scale. The second script needs Pillow,
draw.io Desktop (`drawio`), and `xvfb-run`; it writes `.drawio` sources and SVGs,
plus PNG previews in `/tmp/scarf-page-qa/`.
