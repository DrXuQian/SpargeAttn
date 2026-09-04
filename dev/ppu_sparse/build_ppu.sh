#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${PPU_SDK:-${PPU_HOME:-}}"
python_bin="${PYTHON_BIN:-python}"
out="${OUT:-/workspace/spargeattn-ppu-local-build}"

if [[ -z "$sdk" || ! -x "$sdk/bin/hgcc" || ! -x "$sdk/bin/hgobjdump" ]]; then
  echo '[PPU Sparge build] SKIP: set PPU_SDK to an SDK with hgcc and hgobjdump' >&2
  exit 3
fi
mkdir -p "$out"
PPU_SDK="$sdk" "$python_bin" "$repo/setup_ppu.py" build_ext --force \
  --build-temp "$out/temp" --build-lib "$out/lib"
artifact="$(find "$out/lib/spas_sage_attn" -maxdepth 1 -type f \
  -name '_qattn_ppu*.so' -print -quit)"
if [[ -z "$artifact" ]]; then
  echo '[PPU Sparge build] FAIL: linked extension is missing' >&2
  exit 1
fi
"$python_bin" "$repo/dev/ppu_sparse/check_radial_artifact.py" \
  --artifact "$artifact" --hgobjdump "$sdk/bin/hgobjdump"
sha256sum "$artifact"
printf '[PPU Sparge build] PASS: artifact=%s\n' "$artifact"
