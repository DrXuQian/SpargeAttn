"""Community-compatible H3 top-k and Sol-Attn routing policies."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .plan import SparseAttentionPlan, build_plan


H3_ROUTE_BLOCK = 128
SOL_ROUTE_BLOCK = 64


def _to_bthd(x: torch.Tensor, layout: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Q/K/V must be rank-4")
    if layout == "NHD":
        return x
    if layout == "HND":
        return x.transpose(1, 2)
    raise ValueError("tensor_layout must be HND or NHD")


def _block_counts(length: int, block: int, device: torch.device) -> torch.Tensor:
    blocks = (length + block - 1) // block
    counts = torch.full((blocks,), block, dtype=torch.float32, device=device)
    counts[-1] = length - (blocks - 1) * block
    return counts


def _block_means_bthd(x: torch.Tensor, block: int) -> torch.Tensor:
    """Return contiguous [B,H,Blocks,D] means with an exact tail divisor."""
    batch, length, heads, dim = x.shape
    blocks = (length + block - 1) // block
    padding = blocks * block - length
    padded = F.pad(x, (0, 0, 0, 0, 0, padding))
    sums = padded.reshape(batch, blocks, block, heads, dim).float().sum(2)
    counts = _block_counts(length, block, x.device)
    return (sums / counts[None, :, None, None]).permute(0, 2, 1, 3).contiguous()


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str):
    qb, kb, vb = (_to_bthd(x, layout) for x in (q, k, v))
    if qb.shape[0] != kb.shape[0] or kb.shape != vb.shape:
        raise ValueError("K/V must share shape and Q/K/V batches must match")
    if qb.shape[-1] != kb.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if qb.shape[2] % kb.shape[2]:
        raise ValueError("query heads must be divisible by key/value heads")
    if qb.device != kb.device or qb.device != vb.device:
        raise ValueError("Q/K/V must share one device")
    return qb, kb, vb


def _expand_kv_heads(means: torch.Tensor, query_heads: int) -> torch.Tensor:
    groups = query_heads // means.shape[1]
    return means.repeat_interleave(groups, dim=1) if groups != 1 else means


def make_h3_topk_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    top_k: int,
    tensor_layout: str = "NHD",
    taylor: bool = True,
) -> SparseAttentionPlan:
    """Reproduce h3-sparse-attn's block-mean top-k selection.

    Selected route blocks are canonicalized into ascending order before CSR
    emission.  Selection is unchanged; canonical order makes accumulation and
    repeated-launch fingerprints deterministic.
    """
    qb, kb, vb = _validate_qkv(q, k, v, tensor_layout)
    q_means = _block_means_bthd(qb, H3_ROUTE_BLOCK)
    k_means = _block_means_bthd(kb, H3_ROUTE_BLOCK)
    v_means = _block_means_bthd(vb, H3_ROUTE_BLOCK)
    routed_k = _expand_kv_heads(k_means, qb.shape[2])
    scores = torch.matmul(q_means, routed_k.transpose(-1, -2))
    route_blocks = scores.shape[-1]
    if top_k <= 0 or top_k > route_blocks:
        raise ValueError(f"top_k must be in [1,{route_blocks}], got {top_k}")
    selected = scores.topk(top_k, dim=-1).indices.sort(dim=-1).values
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return build_plan(
        selected_route_mask=mask,
        key_mean=k_means,
        value_mean=v_means,
        block_counts=_block_counts(kb.shape[1], H3_ROUTE_BLOCK, kb.device),
        query_length=qb.shape[1],
        kv_length=kb.shape[1],
        query_block=H3_ROUTE_BLOCK,
        route_block=H3_ROUTE_BLOCK,
        algorithm="h3-topk-summary" if taylor else "h3-topk-hard",
        use_summary=taylor,
    )


def make_sol_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = 1.0,
    thresh_type: str = "diag",
    tensor_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    sink_blocks: Tuple[int, int] = (0, 0),
    sink_query_blocks: Tuple[int, int] = (0, 0),
) -> SparseAttentionPlan:
    """Build Sol-Attn's threshold/neighbor/sink route as canonical CSR."""
    qb, kb, vb = _validate_qkv(q, k, v, tensor_layout)
    if qb.shape[1] != kb.shape[1]:
        raise ValueError("the first PPU Sol path requires equal Q/K sequence lengths")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    q_means = _block_means_bthd(qb, SOL_ROUTE_BLOCK)
    k_means = _block_means_bthd(kb, SOL_ROUTE_BLOCK)
    v_means = _block_means_bthd(vb, SOL_ROUTE_BLOCK)
    routed_k = _expand_kv_heads(k_means, qb.shape[2])
    scale_log2 = (qb.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)) / math.log(2.0)
    scores = torch.matmul(q_means.float(), routed_k.float().transpose(-1, -2)) * scale_log2
    if thresh_type == "exact":
        threshold = scores.mean(-1) + float(tau) * torch.sqrt(
            scores.var(-1, unbiased=False).clamp_min(0.0) + 1.0e-6
        )
    else:
        stat_mean = routed_k.float().mean(2)
        stat_var = (routed_k.float() - stat_mean[:, :, None, :]).square().mean(2)
        threshold_mean = (q_means.float() * stat_mean[:, :, None, :]).sum(-1) * scale_log2
        threshold_var = (
            q_means.float().square() * stat_var[:, :, None, :]
        ).sum(-1) * (scale_log2 * scale_log2)
        threshold = threshold_mean + float(tau) * torch.sqrt(threshold_var.clamp_min(0.0) + 1.0e-6)

    mask = scores > threshold[..., None]
    q_ids = torch.arange(scores.shape[-2], device=scores.device)[:, None]
    k_ids = torch.arange(scores.shape[-1], device=scores.device)[None, :]
    mask |= (q_ids - k_ids).abs()[None, None] <= 1

    sink_begin, sink_end = map(int, sink_blocks)
    sink_q_begin, sink_q_end = map(int, sink_query_blocks)
    if not (0 <= sink_begin <= sink_end <= scores.shape[-1]):
        raise ValueError("sink_blocks is outside the route-block range")
    if not (0 <= sink_q_begin <= sink_q_end <= scores.shape[-2]):
        raise ValueError("sink_query_blocks is outside the query-block range")
    if sink_begin != sink_end:
        mask[..., sink_begin:sink_end] = True
    if sink_q_begin != sink_q_end:
        mask[:, :, sink_q_begin:sink_q_end, :] = True

    return build_plan(
        selected_route_mask=mask,
        key_mean=k_means,
        value_mean=v_means,
        block_counts=_block_counts(kb.shape[1], SOL_ROUTE_BLOCK, kb.device),
        query_length=qb.shape[1],
        kv_length=kb.shape[1],
        query_block=SOL_ROUTE_BLOCK,
        route_block=SOL_ROUTE_BLOCK,
        algorithm=f"sol-{thresh_type}-summary",
        use_summary=True,
    )


__all__ = ["make_h3_topk_plan", "make_sol_plan"]
