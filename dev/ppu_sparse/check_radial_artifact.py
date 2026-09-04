#!/usr/bin/env python3
"""Fail-closed generated-code admission for the PPU Radial specialization."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


MANGLED = (
    "_ZN15spargeattention3ppu6radial20radial_sparse_kernel"
    "EPKaS3_PKN7cutlass6half_tEPNS4_10bfloat16_tEPKfSB_PKiSD_iiiif"
)
DEMANGLED = "spargeattention::ppu::radial::radial_sparse_kernel"


def run(*command: str) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--hgobjdump", type=Path, required=True)
    args = parser.parse_args()
    if not args.artifact.is_file() or not args.hgobjdump.is_file():
        parser.error("artifact and hgobjdump must be existing files")

    listing = run(
        str(args.hgobjdump), "--list-all", "--demangle", str(args.artifact)
    )
    occurrences = listing.count(DEMANGLED)
    if occurrences != 1:
        raise SystemExit(
            f"[PPU Radial artifact] FAIL: exact kernel occurrences={occurrences}, expected 1"
        )

    resources = run(
        str(args.hgobjdump), "--dump-resource-usage=all", str(args.artifact)
    )
    marker = f"{MANGLED} RESOURCE INFO:"
    begin = resources.find(marker)
    if begin < 0:
        raise SystemExit("[PPU Radial artifact] FAIL: resource record is missing")
    resource = resources[begin : resources.find(".ARGUMENT:", begin)]
    vregs = int(re.search(r"vreg_number:(\d+)", resource).group(1))
    sregs = int(re.search(r"sreg_number:(\d+)", resource).group(1))
    stack = int(re.search(r"STACK SIZE:(\d+)", resource).group(1))
    if vregs > 192 or stack != 0:
        raise SystemExit(
            "[PPU Radial artifact] FAIL: "
            f"resources vregs={vregs} stack={stack}, admitted <=192/0"
        )

    isa = run(
        str(args.hgobjdump),
        f"--dump-function={MANGLED}",
        str(args.artifact),
    )
    counts = {
        "qk_i8_mma": isa.count("v.mma.i32.i8.i8.m16n16k32"),
        "pv_f32_mma": isa.count("v.mma.f32.f16.m16n16k16"),
        "bridge_f16_mma": isa.count("v.mma.f16.f16.m16n16k16"),
        "aiu_load": isa.count("vmem.aiu.ld.tsm"),
    }
    expected = {"qk_i8_mma": 16, "pv_f32_mma": 32, "bridge_f16_mma": 4}
    for name, value in expected.items():
        if counts[name] != value:
            raise SystemExit(
                f"[PPU Radial artifact] FAIL: {name}={counts[name]}, expected {value}"
            )
    if counts["aiu_load"] == 0:
        raise SystemExit("[PPU Radial artifact] FAIL: generated kernel lost AIU loads")
    print(
        "[PPU Radial artifact] PASS: symbol=unique geometry=Q128/KV64/8warp "
        f"vregs={vregs} sregs={sregs} stack={stack} "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
