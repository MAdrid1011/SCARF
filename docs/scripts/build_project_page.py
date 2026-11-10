"""Validate and package only the static project's reachable local assets.

Run `python3 docs/scripts/build_project_page.py --output _site` from the
repository root. The output directory must be absent or empty.
"""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

SOURCE = Path(__file__).resolve().parents[1]
SITE_URL = 'https://madrid1011.github.io/SCARF/'


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ('href', 'src'):
            if attrs.get(name):
                self.urls.append(attrs[name])
        if attrs.get('srcset'):
            self.urls.extend(item.strip().split()[0]
                             for item in attrs['srcset'].split(','))
        if attrs.get('property') == 'og:image' or attrs.get('name') == 'twitter:image':
            self.urls.append(attrs['content'])


def local_asset(url, parent):
    if url.startswith(SITE_URL):
        url = url[len(SITE_URL):]
        parent = SOURCE
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    if parsed.path.startswith('/'):
        raise ValueError(f'Use a relative asset path for GitHub Pages: {url}')
    path = (parent / unquote(parsed.path)).resolve()
    if not path.is_relative_to(SOURCE) or not path.is_file():
        raise ValueError(f'Missing or out-of-tree page asset: {url}')
    return path


def build(output):
    output = output.resolve()
    if output.is_relative_to(SOURCE) or SOURCE.is_relative_to(output):
        raise ValueError('Output must be separate from the source directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output must be empty: {output}')
    pending = [SOURCE / 'index.html']
    assets = set()
    while pending:
        path = pending.pop()
        if path in assets:
            continue
        assets.add(path)
        urls = []
        if path.suffix == '.html':
            parser = AssetParser()
            parser.feed(path.read_text())
            urls = parser.urls
        elif path.suffix == '.css':
            urls = re.findall(r'url\(\s*[\'\"]?([^\'\"\)\s]+)', path.read_text())
        for url in urls:
            asset = local_asset(url, path.parent)
            if asset:
                pending.append(asset)
    for path in sorted(assets):
        dest = output / path.relative_to(SOURCE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    (output / '.nojekyll').touch()
    size = sum(path.stat().st_size for path in assets)
    print(f'Packaged {len(assets)} files ({size / 2**20:.2f} MiB) into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('_site'))
    build(parser.parse_args().output)
