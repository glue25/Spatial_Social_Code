"""Verify the release manifest and Python syntax without running experiments."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    failures = []
    python_files = 0
    for name, expected in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            failures.append(f'Unsafe manifest path: {name}')
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f'Missing file: {name}')
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            failures.append(f'Changed file: {name}')
        if path.suffix == '.py':
            python_files += 1
            try:
                ast.parse(content, filename=name)
            except (SyntaxError, ValueError) as error:
                failures.append(f'Invalid Python: {name}: {error}')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        print('Review intentional changes before running scripts/update_checksums.py.', file=sys.stderr)
        return 1
    print(f'PASS: {len(manifest)} file hashes; {python_files} Python files parsed. No model calls made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
