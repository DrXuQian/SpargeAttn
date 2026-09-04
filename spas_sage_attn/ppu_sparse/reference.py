"""Independent PyTorch oracle for a canonical :class:`SparseAttentionPlan`."""

from __future__ import annotations

import math

import torch

from .plan import SparseAttentionPlan, unpack_bits


def _evaluate_plan(
    q_exact: torch.Tensor,
    k_exact: torch.Tensor,
    q_summary: torch.Tensor,
    v: torch.Tensor,
    plan: SparseAttentionPlan,
    *,
    tensor_layout: str = "NHD",
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    """Evaluate exact-token and block-mean contributions explicitly.

    This deliberately does not call the PPU extension or reuse its iteration
    code.  It is slow and intended only for admission fixtures.
    """
    plan.validate(deep=True)
    if tensor_layout == "HND":
        qh, kh, qsh, vh = q_exact, k_exact, q_summary, v
    elif tensor_layout == "NHD":
        qh, kh, qsh, vh = (
            x.transpose(1, 2) for x in (q_exact, k_exact, q_summary, v)
        )
    else:
        raise ValueError("tensor_layout must be HND or NHD")
    scale = q_exact.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)
    output = torch.empty_like(qh)
    lse = torch.empty(
        (plan.batch, plan.query_heads, plan.query_length),
        dtype=torch.float32,
        device=q_exact.device,
    )
    row_ptr = plan.exact_row_ptr.to(torch.int64)
    selected = unpack_bits(plan.selected_route_bits, plan.route_blocks)
    groups = plan.query_heads // plan.kv_heads
    for batch in range(plan.batch):
        for head in range(plan.query_heads):
            kv_head = head // groups
            for qb in range(plan.query_blocks):
                row = (batch * plan.query_heads + head) * plan.query_blocks + qb
                q0 = qb * plan.query_block
                q1 = min(q0 + plan.query_block, plan.query_length)
                begin = int(row_ptr[row].item())
                end = int(row_ptr[row + 1].item())
                exact_blocks = plan.exact_kv64[begin:end].to(torch.int64)
                token_parts = []
                for block in exact_blocks.tolist():
                    k0 = block * 64
                    k1 = min(k0 + 64, plan.kv_length)
                    token_parts.append(
                        torch.arange(k0, k1, device=q_exact.device)
                    )
                logits_parts = []
                values_parts = []
                q_rows = qh[batch, head, q0:q1].float()
                if token_parts:
                    tokens = torch.cat(token_parts)
                    exact_k = kh[batch, kv_head, tokens].float()
                    logits_parts.append(q_rows @ exact_k.transpose(0, 1) * scale)
                    values_parts.append(vh[batch, kv_head, tokens].float())
                if plan.use_summary:
                    approximate = (~selected[row]).nonzero(as_tuple=False).flatten()
                    if approximate.numel():
                        summary_k = plan.key_mean[batch, kv_head, approximate].float()
                        summary_logits = (
                            qsh[batch, head, q0:q1].float()
                            @ summary_k.transpose(0, 1)
                            * scale
                        )
                        summary_logits += plan.log2_block_counts[approximate].mul(math.log(2.0))[None]
                        logits_parts.append(summary_logits)
                        values_parts.append(plan.value_mean[batch, kv_head, approximate].float())
                if not logits_parts:
                    raise ValueError(f"plan row {row} has no exact or summary contribution")
                logits = torch.cat(logits_parts, dim=-1)
                values = torch.cat(values_parts, dim=0)
                probabilities = torch.softmax(logits, dim=-1)
                output[batch, head, q0:q1] = (probabilities @ values).to(output.dtype)
                lse[batch, head, q0:q1] = torch.logsumexp(logits, dim=-1)
    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return (output, lse) if return_lse else output


def sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SparseAttentionPlan,
    *,
    tensor_layout: str = "NHD",
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    return _evaluate_plan(
        q,
        k,
        q,
        v,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        return_lse=return_lse,
    )


def quantized_sparse_attention_reference(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    query_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    plan: SparseAttentionPlan,
    *,
    tensor_layout: str = "NHD",
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    """Oracle for the exact numerical operands consumed by the PPU kernel."""
    if tensor_layout == "NHD":
        q_len, q_heads = query_int8.shape[1:3]
        kv_len, kv_heads = key_int8.shape[1:3]
        qh = query_int8.transpose(1, 2).float()
        kh = key_int8.transpose(1, 2).float()
    elif tensor_layout == "HND":
        q_heads, q_len = query_int8.shape[1:3]
        kv_heads, kv_len = key_int8.shape[1:3]
        qh = query_int8.float()
        kh = key_int8.float()
    else:
        raise ValueError("tensor_layout must be HND or NHD")
    q_scale_index = torch.arange(q_len, device=query_int8.device) // 32
    k_scale_index = torch.arange(kv_len, device=key_int8.device) // 64
    qh = qh * query_scale[:, :, q_scale_index, None]
    kh = kh * key_scale[:, :, k_scale_index, None]
    if tensor_layout == "NHD":
        q_dequant = qh.transpose(1, 2)
        k_dequant = kh.transpose(1, 2)
    else:
        q_dequant, k_dequant = qh, kh
    return _evaluate_plan(
        q_dequant,
        k_dequant,
        query_fp16,
        value_fp16,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        return_lse=return_lse,
    )


__all__ = [
    "quantized_sparse_attention_reference",
    "sparse_attention_reference",
]
