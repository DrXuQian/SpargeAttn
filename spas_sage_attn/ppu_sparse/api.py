"""Public PPU SpargeAttention forward APIs."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .plan import SparseAttentionPlan
from .planners import make_h3_topk_plan, make_sol_plan
from .radial import (
    RadialAttentionPlan,
    make_radial_plan_from_compute_mask,
)


def _shape_contract(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str):
    if layout == "NHD":
        batch, q_len, q_heads, head_dim = q.shape
        kb, kv_len, kv_heads, kd = k.shape
        vb, vv_len, vv_heads, vd = v.shape
        seq_dim = 1
    elif layout == "HND":
        batch, q_heads, q_len, head_dim = q.shape
        kb, kv_heads, kv_len, kd = k.shape
        vb, vv_heads, vv_len, vd = v.shape
        seq_dim = 2
    else:
        raise ValueError("tensor_layout must be HND or NHD")
    if (kb, vb) != (batch, batch) or (kv_len, kv_heads, kd) != (vv_len, vv_heads, vd):
        raise ValueError("Q/K/V batch or K/V shape contract is invalid")
    if kd != head_dim or q_heads % kv_heads:
        raise ValueError("Q/K head dimensions or GQA head grouping are invalid")
    return batch, q_len, kv_len, q_heads, kv_heads, head_dim, seq_dim


def spargeattn_block_sparse_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SparseAttentionPlan,
    *,
    tensor_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    return_lse: bool = False,
):
    """Execute a validated exact-CSR/summary plan on the PPU Sparge core."""
    try:
        from .. import ppu_compile
    except ImportError as error:
        raise RuntimeError("PPU SpargeAttention extension is not installed") from error

    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("the first PPU Sparse-Sage path requires bf16 Q/K/V")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("PPU Sparse-Sage Q/K/V dtypes must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("PPU Sparse-Sage Q/K/V must share one device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("PPU Sparse-Sage Q/K/V must be rank-4")
    batch, q_len, kv_len, q_heads, kv_heads, head_dim, seq_dim = _shape_contract(
        q, k, v, tensor_layout
    )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("PPU Sparse-Sage requires contiguous head dimensions")
    if (
        plan.batch != batch
        or plan.query_length != q_len
        or plan.kv_length != kv_len
        or plan.query_heads != q_heads
        or plan.kv_heads != kv_heads
        or plan.head_dim != head_dim
    ):
        raise ValueError("sparse plan identity does not match Q/K/V")
    plan.validate(deep=False)
    if not plan.admitted:
        raise ValueError(
            "sparse plan was not produced by an admitted planner; "
            "call plan.validate(deep=True) once before execution"
        )
    if plan.exact_row_ptr.device != q.device:
        raise ValueError("sparse plan and Q/K/V must share one device")
    if head_dim != 128:
        raise ValueError("the first PPU Sparse-Sage path requires head_dim 128")
    if return_lse:
        raise ValueError("the first PPU Sparse-Sage path does not publish LSE")
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    km = k.mean(dim=seq_dim) if smooth_k else None
    q_int8, q_scale, k_int8, k_scale = ppu_compile.quant_per_warp_int8(
        q, k, km, tensor_layout=tensor_layout
    )
    q_fp16 = q.to(torch.float16).contiguous()
    v_fp16 = v.to(torch.float16).contiguous()
    key_mean = plan.key_mean
    if km is not None:
        # K centering shifts every exact token and summary by the same vector;
        # routing is invariant but the executor must use the centered summary.
        key_mean = (key_mean - km.to(torch.float16)[:, :, None, :]).contiguous()
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    layout_id = 0 if tensor_layout == "NHD" else 1
    lse = ppu_compile.qk_int8_sv_f16_block_sparse_accum_f32_attn(
        q_int8,
        k_int8,
        v_fp16,
        q_fp16,
        output,
        q_scale,
        k_scale,
        plan.exact_row_ptr,
        plan.exact_kv64,
        plan.selected_route_bits,
        key_mean,
        plan.value_mean,
        plan.log2_block_counts,
        layout_id,
        plan.query_block,
        plan.route_block,
        int(plan.use_summary),
        float(sm_scale),
        int(return_lse),
    )
    if return_lse:
        lse = lse / 1.4426950408889634
        if km is not None:
            if tensor_layout == "NHD":
                qh = q.transpose(1, 2)
            else:
                qh = q
            km_q = km.repeat_interleave(q_heads // kv_heads, dim=1)
            correction = torch.matmul(
                qh, km_q.unsqueeze(-1)
            ).squeeze(-1).float()
            lse = lse + correction * float(sm_scale)
        return output, lse
    return output


def spargeattn_radial_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: RadialAttentionPlan,
    *,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
):
    """Run the dedicated PPU SparseSage2/Radial forward executor.

    Radial owns mask policy.  This operator starts at its Q128/KV64
    incremental-LUT boundary and deliberately does not route through the
    generic H3/Sol summary executor.
    """
    try:
        from .. import ppu_compile
    except ImportError as error:
        raise RuntimeError("PPU SpargeAttention extension is not installed") from error

    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("the first PPU Radial path requires bf16 Q/K/V")
    if q.device != k.device or q.device != v.device:
        raise ValueError("PPU Radial Q/K/V must share one device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("PPU Radial Q/K/V must be rank-4")
    batch, q_len, kv_len, q_heads, kv_heads, head_dim, seq_dim = _shape_contract(
        q, k, v, tensor_layout
    )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("PPU Radial requires contiguous head dimensions")
    identity = (
        plan.batch,
        plan.query_length,
        plan.kv_length,
        plan.query_heads,
        plan.kv_heads,
        plan.head_dim,
    )
    if identity != (batch, q_len, kv_len, q_heads, kv_heads, head_dim):
        raise ValueError("Radial plan identity does not match Q/K/V")
    plan.validate(deep=False)
    if not plan.admitted:
        raise ValueError(
            "Radial plan was not deeply admitted; call plan.validate(deep=True)"
        )
    if plan.block_lut.device != q.device:
        raise ValueError("Radial plan and Q/K/V must share one device")
    if return_lse:
        raise ValueError("the first PPU Radial path does not publish LSE")
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    # Subtracting the same mean from every K leaves softmax probabilities and
    # O unchanged.  With no LSE publication, no host correction is needed.
    km = k.mean(dim=seq_dim) if smooth_k else None
    q_int8, q_scale, k_int8, k_scale = ppu_compile.quant_per_warp_int8(
        q, k, km, tensor_layout=tensor_layout
    )
    value_fp16 = v.to(torch.float16).contiguous()
    output = torch.empty_like(q)
    ppu_compile.qk_int8_sv_f16_radial_accum_f32_attn(
        q_int8,
        k_int8,
        value_fp16,
        output,
        q_scale,
        k_scale,
        plan.block_lut,
        plan.valid_block_num,
        0 if tensor_layout == "NHD" else 1,
        float(sm_scale),
    )
    return output


def block_sparse_sage2_attn_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_id: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    smooth_k: bool = True,
    tensor_layout: str = "HND",
    return_sparsity: bool = False,
):
    """PPU adapter for Radial's existing SparseSage2 operator boundary.

    ``mask_id`` is Radial's already-converted non-SM90 Q128/KV64 mask.  Mask
    generation and dense-layer/timestep policy remain owned by Radial.
    """
    if dropout_p != 0:
        raise ValueError("PPU Radial forward does not support dropout")
    batch, q_len, kv_len, q_heads, kv_heads, head_dim, _ = _shape_contract(
        q, k, v, tensor_layout
    )
    plan = make_radial_plan_from_compute_mask(
        mask_id,
        batch=batch,
        query_heads=q_heads,
        kv_heads=kv_heads,
        query_length=q_len,
        kv_length=kv_len,
        head_dim=head_dim,
    )
    output = spargeattn_radial_ppu(
        q,
        k,
        v,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=scale,
        smooth_k=smooth_k,
    )
    if return_sparsity:
        sparsity = 1.0 - plan.selected_tiles / plan.block_lut.numel()
        return output, sparsity
    return output


def spargeattn_h3_topk_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    top_k: int = 64,
    tensor_layout: str = "NHD",
    taylor: bool = True,
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    return_lse: bool = False,
    return_plan: bool = False,
):
    plan = make_h3_topk_plan(
        q, k, v, top_k=top_k, tensor_layout=tensor_layout, taylor=taylor
    )
    result = spargeattn_block_sparse_ppu(
        q,
        k,
        v,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )
    return (result, plan) if return_plan else result


def spargeattn_sol_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = 1.0,
    thresh_type: str = "diag",
    tensor_layout: str = "NHD",
    sink_blocks: Tuple[int, int] = (0, 0),
    sink_query_blocks: Tuple[int, int] = (0, 0),
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    return_lse: bool = False,
    return_plan: bool = False,
):
    plan = make_sol_plan(
        q,
        k,
        v,
        tau=tau,
        thresh_type=thresh_type,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        sink_blocks=sink_blocks,
        sink_query_blocks=sink_query_blocks,
    )
    result = spargeattn_block_sparse_ppu(
        q,
        k,
        v,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )
    return (result, plan) if return_plan else result


__all__ = [
    "block_sparse_sage2_attn_ppu",
    "spargeattn_block_sparse_ppu",
    "spargeattn_h3_topk_ppu",
    "spargeattn_radial_ppu",
    "spargeattn_sol_ppu",
]
