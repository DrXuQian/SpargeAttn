#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sage_repo="${SAGEATTENTION_REPO:-$(cd "$repo/.." && pwd)/SageAttention}"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/spargeattn-ppu-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
runtime_dir="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"

if [[ ! -d "$sage_repo/.git" ]]; then
  printf '[PPU Sparge box] FAIL: dense SageAttention anchor repo missing: %s\n' \
    "$sage_repo" >&2
  exit 1
fi
printf '[PPU Sparge box] mode=EXECUTION-ONLY sha=%s sage=%s runtime=%s out=%s\n' \
  "$sha" "$(git -C "$sage_repo" rev-parse HEAD)" "$runtime_dir" "$out"

python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" --prebuilt-root "$repo/prebuilt/ppu_10" \
  --runtime-dir "$runtime_dir" --json-out "$out/sparge-prebuilt.json" \
  --artifact-path-out "$out/sparge-artifact.path" \
  2>&1 | tee "$out/sparge-prebuilt.log"
python "$sage_repo/tools/verify_ppu_prebuilt.py" \
  --repo "$sage_repo" --prebuilt-root "$sage_repo/prebuilt/ppu_10" \
  --runtime-dir "$runtime_dir" --json-out "$out/sage-prebuilt.json" \
  --artifact-path-out "$out/sage-artifact.path" \
  2>&1 | tee "$out/sage-prebuilt.log"

sparge_extension="$(<"$out/sparge-artifact.path")"
sage_extension="$(<"$out/sage-artifact.path")"
cp -f "$sparge_extension" "$repo/spas_sage_attn/"
cp -f "$sage_extension" "$sage_repo/sageattention/"
sha256sum "$sparge_extension" "$sage_extension" | tee "$out/binaries.sha256"

export LD_LIBRARY_PATH="$runtime_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:$sage_repo"
python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/host-plan-admission.log"
python "$repo/dev/ppu_sparse/device_smoke.py" 2>&1 | tee "$out/device-smoke.log"

if [[ "${RUN_PERF:-1}" == 1 ]]; then
  perf_args=(
    --batch "${BATCH:-1}"
    --heads "${HEADS:-16}"
    --kv-heads "${KV_HEADS:-16}"
    --seq "${SEQ:-4096}"
    --top-k "${TOP_K:-8}"
    --tau "${TAU:-1.0}"
    --warmup "${WARMUP:-3}"
    --samples "${SAMPLES:-7}"
    --launches "${LAUNCHES:-10}"
  )
  if [[ -n "${RADIAL_MASK:-}" ]]; then
    perf_args+=(--radial-mask "$RADIAL_MASK"
      --radial-mask-kind "${RADIAL_MASK_KIND:-compute}")
  fi
  python "$repo/dev/ppu_sparse/device_perf.py" "${perf_args[@]}" \
    2>&1 | tee "$out/device-perf.log"
fi

printf '[PPU Sparge box] PASS: artifacts=%s\n' "$out"
