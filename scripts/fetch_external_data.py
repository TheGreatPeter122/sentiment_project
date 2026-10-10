"""Download two pinned public CSV sources into ignored data/raw/."""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a0b781d72158bde242c0626c9e354d2b328865ff'
BASE = f'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/{REVISION}/datasets'
SOURCES = {
    'online_shopping_10_cats.zip': (
        f'{BASE}/online_shopping_10_cats/online_shopping_10_cats.zip',
        '6816ded66fde9d0927ecbd370b5e12d0174d1abac35192ec014ac2a0a8d25858',
    ),
    'waimai_10k.csv': (
        f'{BASE}/waimai_10k/waimai_10k.csv',
        'b8b5c11c00b51578d35937fca56e03a126b45381f1ea4b22893237cb377a26fe',
    ),
}


def main():
    target = ROOT / 'data/raw'
    target.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = target / name
        if not path.exists():
            print(f'Downloading {name}...')
            with urllib.request.urlopen(url, timeout=60) as response:
                contents = response.read()
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(f'{name}: downloaded SHA256 mismatch')
            path.write_bytes(contents)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: local SHA256 mismatch')
        print(f'{name}: verified')


if __name__ == '__main__':
    main()
