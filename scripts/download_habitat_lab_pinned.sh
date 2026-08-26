#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REV=094d6be2f9d057e4781a68ae792132895fd4d3d0
ARCHIVE=/tmp/habitat-lab-${REV}.complete.tar.gz
PART=/tmp/habitat-lab-${REV}.attempt.tar.gz
LOG=workdirs/habitat_enacttom_setup.log
STATUS=workdirs/habitat_enacttom_setup.status
URL="https://codeload.github.com/facebookresearch/habitat-lab/tar.gz/${REV}"

date -u '+download_started=%Y-%m-%dT%H:%M:%SZ' > "$STATUS"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "habitat_lab_download_attempt=$attempt" >> "$LOG"
  if curl --http1.1 -fL --connect-timeout 20 --max-time 1800 "$URL" -o "$PART" >> "$LOG" 2>&1 && gzip -t "$PART"; then
    mv "$PART" "$ARCHIVE"
    break
  fi
done

gzip -t "$ARCHIVE"
tar -xzf "$ARCHIVE" -C external
mv "external/habitat-lab-${REV}" external/habitat-lab
printf '%s\n' "$REV" > external/habitat-lab/PINNED_COMMIT
date -u '+download_completed=%Y-%m-%dT%H:%M:%SZ' >> "$STATUS"
