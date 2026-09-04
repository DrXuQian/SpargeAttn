#!/usr/bin/env python3
"""Dense/H3-top-k/Sol performance decomposition on one PPU device."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics

import torch

from spas_sage_attn import ppu_compile
from sageattention import ppu_compile as sage_ppu_compile
from sageattention.core import sageattn_qk_int8_pv_fp16_ppu
from spas_sage_attn.ppu_sparse import (
    make_radial_plan_from_block_mask,
    make_radial_plan_from_compute_mask,
    make_h3_topk_plan,
    make_sol_plan,
    spargeattn_radial_ppu,
    spargeattn_block_sparse_ppu,
)
from spas_sage_attn.ppu_sparse.plan import unpack_bits


def measure_us(fn, warmup: int, samples: int, launches: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(launches):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(begin.elapsed_time(end) * 1000.0 / launches)
    return values


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def report(role: str, values: list[float], dense_flops: int, output, extra: str = ""):
    median = statistics.median(values)
    effective = dense_flops / median / 1.0e6
    print(
        f"[PPU sparse perf] role={role} median_us={median:.3f} "
        f"range=[{min(values):.3f},{max(values):.3f}] "
        f"dense_equivalent_tflops={effective:.3f} output_sha256={digest(output)} {extra}".rstrip()
    )
    return median


def plan_density(plan) -> tuple[int, int, int]:
    row_ptr = plan.exact_row_ptr.cpu().to(torch.int64)
    exact_tiles = int(plan.exact_kv64.numel())
    full_tiles = plan.rows * plan.kv_blocks
    selected = unpack_bits(plan.selected_route_bits, plan.route_blocks)
    summaries = int((~selected).sum().item()) if plan.use_summary else 0
    assert row_ptr[0].item() == 0 and row_ptr[-1].item() == exact_tiles
    return exact_tiles, full_tiles, summaries


def make_core(q, k, v, plan):
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(
        q, k, None, tensor_layout="NHD"
    )
    qf = q.to(torch.float16).contiguous()
    vf = v.to(torch.float16).contiguous()
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    def core():
        ppu_compile.qk_int8_sv_f16_block_sparse_accum_f32_attn(
            qi,
            ki,
            vf,
            qf,
            output,
            qs,
            ks,
            plan.exact_row_ptr,
            plan.exact_kv64,
            plan.selected_route_bits,
            plan.key_mean,
            plan.value_mean,
            plan.log2_block_counts,
            0,
            plan.query_block,
            plan.route_block,
            int(plan.use_summary),
            q.shape[-1] ** -0.5,
            0,
        )

    return core, output


def make_radial_fixture(seq: int, device: torch.device) -> torch.Tensor:
    """Irregular operator fixture; Radial/Wan policy remains external."""
    blocks = math.ceil(seq / 128)
    mask = torch.zeros((blocks, blocks), dtype=torch.bool, device=device)
    for row in range(blocks):
        radius = 1 + row % 4
        lo, hi = max(0, row - radius), min(blocks, row + radius + 1)
        mask[row, lo:hi] = True
        mask[row, 0] = True
        if row % 3 == 0:
            mask[row, blocks - 1] = True
    return mask


def make_radial_core(q, k, v, plan):
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(
        q, k, None, tensor_layout="NHD"
    )
    vf = v.to(torch.float16).contiguous()
    output = torch.empty_like(q)

    def core():
        ppu_compile.qk_int8_sv_f16_radial_accum_f32_attn(
            qi,
            ki,
            vf,
            output,
            qs,
            ks,
            plan.block_lut,
            plan.valid_block_num,
            0,
            q.shape[-1] ** -0.5,
        )

    return core, output


def make_dense_core(q, k, v):
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(
        q, k, None, tensor_layout="NHD"
    )
    vf = v.to(torch.float16).contiguous()
    output = torch.empty_like(q)

    def core():
        sage_ppu_compile.qk_int8_sv_f16_accum_f32_attn(
            qi,
            ki,
            vf,
            output,
            qs,
            ks,
            0,
            0,
            2,
            q.shape[-1] ** -0.5,
            0,
        )

    return core, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--radial-mask", type=str)
    parser.add_argument(
        "--radial-mask-kind",
        choices=("compute", "source64", "source128"),
        default="compute",
    )
    args = parser.parse_args()
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        parser.error("heads must be positive and divisible by kv-heads")
    route_blocks = math.ceil(args.seq / 128)
    if not (0 < args.top_k <= route_blocks):
        parser.error(f"top-k must be in [1,{route_blocks}]")

    torch.cuda.set_device(0)
    torch.manual_seed(0x5A6E)
    q = torch.randn(
        (args.batch, args.seq, args.heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    k = torch.randn(
        (args.batch, args.seq, args.kv_heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    v = torch.randn_like(k) * 0.2
    dense_output = torch.empty_like(q)
    h3_output = torch.empty_like(q)
    sol_output = torch.empty_like(q)
    radial_output = torch.empty_like(q)

    def dense():
        nonlocal dense_output
        dense_output = sageattn_qk_int8_pv_fp16_ppu(
            q, k, v, tensor_layout="NHD", smooth_k=False
        )

    h3_plan = make_h3_topk_plan(
        q, k, v, top_k=args.top_k, tensor_layout="NHD", taylor=True
    )
    sol_plan = make_sol_plan(
        q, k, v, tau=args.tau, tensor_layout="NHD", thresh_type="diag"
    )
    h3_core, h3_core_output = make_core(q, k, v, h3_plan)
    sol_core, sol_core_output = make_core(q, k, v, sol_plan)
    if args.radial_mask:
        radial_source = torch.load(
            args.radial_mask, map_location=q.device, weights_only=True
        )
        if not isinstance(radial_source, torch.Tensor):
            raise TypeError("--radial-mask must contain one Tensor")
        if args.radial_mask_kind == "compute":
            radial_plan = make_radial_plan_from_compute_mask(
                radial_source,
                batch=args.batch,
                query_heads=args.heads,
                kv_heads=args.kv_heads,
                query_length=args.seq,
                kv_length=args.seq,
            )
        else:
            radial_plan = make_radial_plan_from_block_mask(
                radial_source,
                batch=args.batch,
                query_heads=args.heads,
                kv_heads=args.kv_heads,
                query_length=args.seq,
                kv_length=args.seq,
                source_block=64 if args.radial_mask_kind == "source64" else 128,
            )
        radial_mask_label = f"file:{args.radial_mask_kind}"
    else:
        radial_source = make_radial_fixture(args.seq, q.device)
        radial_plan = make_radial_plan_from_block_mask(
            radial_source,
            batch=args.batch,
            query_heads=args.heads,
            kv_heads=args.kv_heads,
            query_length=args.seq,
            kv_length=args.seq,
            source_block=128,
        )
        radial_mask_label = "synthetic-nonuniform"
    radial_core, radial_core_output = make_radial_core(q, k, v, radial_plan)
    dense_core, dense_core_output = make_dense_core(q, k, v)

    def h3_preplanned():
        nonlocal h3_output
        h3_output = spargeattn_block_sparse_ppu(
            q, k, v, h3_plan, tensor_layout="NHD"
        )

    def sol_preplanned():
        nonlocal sol_output
        sol_output = spargeattn_block_sparse_ppu(
            q, k, v, sol_plan, tensor_layout="NHD"
        )

    def radial_preplanned():
        nonlocal radial_output
        radial_output = spargeattn_radial_ppu(
            q,
            k,
            v,
            radial_plan,
            tensor_layout="NHD",
            smooth_k=False,
        )

    # Planner timings stand alone: Python/Torch route generation is not hidden
    # inside a core-only headline.  A production H3 call must pay it each time.
    h3_plan_samples = measure_us(
        lambda: make_h3_topk_plan(
            q, k, v, top_k=args.top_k, tensor_layout="NHD", taylor=True
        ),
        1,
        args.samples,
        1,
    )
    sol_plan_samples = measure_us(
        lambda: make_sol_plan(
            q, k, v, tau=args.tau, tensor_layout="NHD", thresh_type="diag"
        ),
        1,
        args.samples,
        1,
    )

    dense_flops = 4 * args.batch * args.heads * args.seq * args.seq * 128
    print(
        f"[PPU sparse perf config] B={args.batch} H={args.heads} Hkv={args.kv_heads} "
        f"N={args.seq} D=128 top_k={args.top_k}/{route_blocks} tau={args.tau} "
        f"warmup={args.warmup} samples={args.samples} launches={args.launches}"
    )
    h3_exact, h3_full, h3_summaries = plan_density(h3_plan)
    sol_exact, sol_full, sol_summaries = plan_density(sol_plan)
    print(
        f"[PPU sparse plan cost] h3_median_us={statistics.median(h3_plan_samples):.3f} "
        f"sol_median_us={statistics.median(sol_plan_samples):.3f} "
        f"h3_exact_kv64={h3_exact}/{h3_full} h3_summary_rows={h3_summaries} "
        f"sol_exact_kv64={sol_exact}/{sol_full} sol_summary_rows={sol_summaries}"
    )

    dense_values = measure_us(dense, args.warmup, args.samples, args.launches)
    dense_core_values = measure_us(
        dense_core, args.warmup, args.samples, args.launches
    )
    h3_core_values = measure_us(h3_core, args.warmup, args.samples, args.launches)
    h3_values = measure_us(h3_preplanned, args.warmup, args.samples, args.launches)
    sol_core_values = measure_us(sol_core, args.warmup, args.samples, args.launches)
    sol_values = measure_us(sol_preplanned, args.warmup, args.samples, args.launches)
    radial_core_values = measure_us(
        radial_core, args.warmup, args.samples, args.launches
    )
    radial_values = measure_us(
        radial_preplanned, args.warmup, args.samples, args.launches
    )
    dense()
    dense_core()
    h3_preplanned()
    sol_preplanned()
    radial_preplanned()
    torch.cuda.synchronize()
    dense_us = report("dense-quant+core", dense_values, dense_flops, dense_output)
    dense_core_us = report(
        "dense-prequantized-core",
        dense_core_values,
        dense_flops,
        dense_core_output,
    )
    h3_core_us = report(
        "h3-prequantized-core",
        h3_core_values,
        dense_flops,
        h3_core_output,
        f"exact_density={h3_exact / h3_full:.6f}",
    )
    h3_us = report(
        "h3-preplanned-quant+core", h3_values, dense_flops, h3_output
    )
    sol_core_us = report(
        "sol-prequantized-core",
        sol_core_values,
        dense_flops,
        sol_core_output,
        f"exact_density={sol_exact / sol_full:.6f}",
    )
    sol_us = report(
        "sol-preplanned-quant+core", sol_values, dense_flops, sol_output
    )
    radial_density = radial_plan.selected_tiles / radial_plan.block_lut.numel()
    radial_core_us = report(
        "radial-prequantized-core",
        radial_core_values,
        dense_flops,
        radial_core_output,
        f"exact_density={radial_density:.6f} mask={radial_mask_label}",
    )
    radial_us = report(
        "radial-preplanned-quant+core",
        radial_values,
        dense_flops,
        radial_output,
        f"exact_density={radial_density:.6f} mask={radial_mask_label}",
    )
    radial_efficiency = (dense_core_us / radial_core_us) * radial_density
    if radial_efficiency >= 0.85:
        radial_alignment = "ALIGNED"
    elif radial_efficiency >= 0.70:
        radial_alignment = "PARTIAL"
    else:
        radial_alignment = "NOT-ALIGNED"
    print(
        f"[PPU sparse perf verdict] h3_core_speedup={dense_us / h3_core_us:.3f}x "
        f"h3_preplanned_speedup={dense_us / h3_us:.3f}x "
        f"h3_with_plan_model={dense_us / (h3_us + statistics.median(h3_plan_samples)):.3f}x "
        f"sol_core_speedup={dense_us / sol_core_us:.3f}x "
        f"sol_preplanned_speedup={dense_us / sol_us:.3f}x "
        f"sol_with_plan_model={dense_us / (sol_us + statistics.median(sol_plan_samples)):.3f}x "
        f"radial_core_speedup={dense_core_us / radial_core_us:.3f}x "
        f"radial_preplanned_speedup={dense_us / radial_us:.3f}x "
        f"radial_density={radial_density:.6f} "
        f"radial_density_efficiency={radial_efficiency:.3f} "
        f"radial_core_alignment={radial_alignment} "
        "thresholds=ALIGNED>=0.85/PARTIAL>=0.70"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
