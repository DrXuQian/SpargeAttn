#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${SPARGEATTN_PPU_OUT:-/workspace/spargeattn-ppu-local}"
mkdir -p "$out"

PYTHONPATH="$repo" python "$repo/dev/ppu_sparse/test_plans.py" \
  | tee "$out/plan-oracle.log"
python -m py_compile "$repo"/spas_sage_attn/ppu_compile.py \
  "$repo"/spas_sage_attn/ppu_sparse/*.py \
  "$repo"/tools/verify_ppu_prebuilt.py

python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" --prebuilt-root "$repo/prebuilt/ppu_10" \
  --json-out "$out/prebuilt-identity.json" --self-test \
  | tee "$out/prebuilt-identity.log"

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

python - "$repo/tools/run_ppu_sparse_box.sh" <<'PY'
from pathlib import Path
import sys

runner = Path(sys.argv[1]).read_text()
forbidden = ("hgcc", "hgobjdump", "build_ext", "setup_ppu.py")
bad = [token for token in forbidden if token in runner]
if bad:
    raise SystemExit(f"[PPU Sparge runner] FAIL: execution runner can compile: {bad}")
if '[[ ! -d "$sage_repo/.git" ]]' in runner:
    raise SystemExit(
        "[PPU Sparge runner] FAIL: Sage repository guard rejects gitfile submodules"
    )
if 'git -C "$sage_repo" rev-parse --is-inside-work-tree' not in runner:
    raise SystemExit(
        "[PPU Sparge runner] FAIL: Sage repository guard is not Git-semantic"
    )
print("[PPU Sparge runner] execution-only/PASS")
PY

printf '[PPU Sparge local] PASS: planner/oracle + prebuilt + ownership + runner\n'
