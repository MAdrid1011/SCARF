"""Export complete web figures from the paper sources without cut-off axes.

Run with Python, PyMuPDF 1.26+, and Pillow. The source PDFs are never saved or
modified. Qualitative panels share the original, unmodified color scale.
"""
from pathlib import Path
import io

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'micro59-camera-ready/figures'
MEDIA = ROOT / 'docs/media'


def render(page, name, clip=None, scale=8):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    image.save(MEDIA / f'{name}.webp', quality=95)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    # The original qualitative PDF has text beyond its MediaBox. Expanding
    # the in-memory page recovers the tick labels before individual crops.
    doc = fitz.open(SOURCE / 'StressCase_latency_error_maps_preview.pdf')
    page = doc[0]
    page.set_mediabox(fitz.Rect(-12, -5, 264, 163))
    panels = {
        'quality-fsdr-metric': (-6, -1, 63, 63),
        'quality-fsdr-latency': (63, -1, 126, 63),
        'quality-fsdr-render': (126.65, 2.2, 177.7, 53.25),
        'quality-fsdr-error': (179.7, 2.2, 230.8, 53.25),
        'quality-saes-metric': (-6, 76.5, 63, 140.5),
        'quality-saes-latency': (63, 76.5, 126, 140.5),
        'quality-saes-render': (126.65, 79.7, 177.7, 130.75),
        'quality-saes-error': (179.7, 79.7, 230.8, 130.75),
        'quality-colorbar': (232, 0, 262, 134),
    }
    for name, (x0, y0, x1, y1) in panels.items():
        render(page, name, fitz.Rect(x0 + 12, y0 + 5, x1 + 12, y1 + 5), 9)

    # Extract the actual scene bitmap rather than a framed PDF screenshot.
    doc = fitz.open(SOURCE / 'Challenge2.drawio.pdf')
    xref = next(item[0] for item in doc[0].get_images() if item[2:4] == (1848, 1848))
    source_image = Image.open(io.BytesIO(doc.extract_image(xref)['image']))
    source_image.resize((768, 768)).save(MEDIA / 'scene-bedroom.webp', quality=95)

    for filename, name in [
        ('Dataflow.drawio.pdf', 'dataflow'),
        ('Arch.drawio.pdf', 'architecture'),
        ('Performance_combined.pdf', 'performance'),
        ('Ablation.drawio.pdf', 'ablation'),
    ]:
        doc = fitz.open(SOURCE / filename)
        page = doc[0]
        bounds = page.rect
        page.set_mediabox(fitz.Rect(-3, -3, bounds.width + 3, bounds.height + 3))
        render(page, name, scale=2600 / page.rect.width)
    print('Exported scene, complete paper diagrams, and qualitative panels.')


if __name__ == '__main__':
    main()
