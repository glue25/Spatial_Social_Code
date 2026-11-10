"""Refresh release hashes from tracked and unignored files; review the diff first."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    )
    names = sorted({name.decode('utf-8') for name in output.split(b'\0') if name})
    manifest = {}
    for name in names:
        if name == 'SHA256SUMS.json':
            continue
        path = ROOT / name
        if path.is_symlink():
            raise ValueError(f'Release manifest does not support symlinks: {name}')
        if path.is_file():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'SHA256SUMS.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f'Updated {len(manifest)} release file hashes. Review SHA256SUMS.json before committing.')


if __name__ == '__main__':
    main()
