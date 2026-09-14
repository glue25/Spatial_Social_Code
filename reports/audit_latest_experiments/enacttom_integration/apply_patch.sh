#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="093c4a11b622c8f0b50070e51ae9461b2b1db2f8"
TARGET="${1:?usage: apply_patch.sh /path/to/EnactTom}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

actual="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$actual" != "$EXPECTED_COMMIT" ]]; then
  echo "refusing: expected EnactToM $EXPECTED_COMMIT, found $actual" >&2
  exit 2
fi
if [[ -n "$(git -C "$TARGET" status --porcelain)" ]]; then
  echo "refusing: target EnactToM worktree is not clean" >&2
  exit 2
fi

git -C "$TARGET" apply --check "$HERE/ssc_closed_loop.patch"
git -C "$TARGET" apply "$HERE/ssc_closed_loop.patch"
echo "Applied SSC integration patch to $TARGET"
