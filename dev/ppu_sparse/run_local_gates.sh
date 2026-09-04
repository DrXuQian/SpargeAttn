#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${SPARGEATTN_PPU_OUT:-/workspace/spargeattn-ppu-local}"
mkdir -p "$out"

PYTHONPATH="$repo" python "$repo/dev/ppu_sparse/test_plans.py" \
  | tee "$out/plan-oracle.log"
python -m py_compile "$repo"/spas_sage_attn/ppu_compile.py \
  "$repo"/spas_sage_attn/ppu_sparse/*.py

python - "$repo" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
sage = root.parent / "SageAttention"
forbidden = {
    "csrc/qattn/ppu/block_sparse_ppu.cu",
    "csrc/qattn/ppu/radial_sparse_ppu.cu",
    "sageattention/ppu_sparse",
    "tools/run_ppu_sparse_box.sh",
}
present = [entry for entry in forbidden if (sage / entry).exists()]
if present:
    raise SystemExit(
        "[PPU Sparge ownership] FAIL: sparse implementation remains in "
        f"SageAttention: {present}"
    )
print("[PPU Sparge ownership] SageAttention sparse implementation absent/PASS")
PY

printf '[PPU Sparge local] PASS: planner/oracle + Python compile + ownership\n'
