"""Independent PyTorch oracles for the PPU Radial LUT contract."""

from __future__ import annotations

import torch

from .radial import RadialAttentionPlan


def _evaluate_radial(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: RadialAttentionPlan,
    *,
    tensor_layout: str,
    sm_scale: float | None,
) -> torch.Tensor:
    """Evaluate LUT rows without calling the device executor or mask recovery."""
    plan.validate(deep=True)
    if tensor_layout == "NHD":
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    elif tensor_layout == "HND":
        qh, kh, vh = q, k, v
    else:
        raise ValueError("tensor_layout must be HND or NHD")
    scale = q.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)
    output = torch.empty_like(qh)
    groups = plan.query_heads // plan.kv_heads
    lut = plan.block_lut.to(torch.int64)
    counts = plan.valid_block_num.to(torch.int64)
    for batch in range(plan.batch):
        for head in range(plan.query_heads):
            kv_head = head // groups
            for qb in range(plan.query_blocks):
                count = int(counts[batch, head, qb].item())
                blocks = lut[batch, head, qb, :count].cumsum(dim=-1)
                token_parts = []
                for block in blocks.tolist():
                    begin = block * 64
                    token_parts.append(
                        torch.arange(
                            begin,
                            min(begin + 64, plan.kv_length),
                            device=q.device,
                        )
                    )
                tokens = torch.cat(token_parts)
                q0 = qb * 128
                q1 = min(q0 + 128, plan.query_length)
                query_rows = qh[batch, head, q0:q1].float()
                keys = kh[batch, kv_head, tokens].float()
                values = vh[batch, kv_head, tokens].float()
                probabilities = torch.softmax(
                    query_rows @ keys.transpose(0, 1) * scale, dim=-1
                )
                output[batch, head, q0:q1] = (
                    probabilities @ values
                ).to(output.dtype)
    return output.transpose(1, 2) if tensor_layout == "NHD" else output


def radial_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: RadialAttentionPlan,
    *,
    tensor_layout: str = "HND",
    sm_scale: float | None = None,
) -> torch.Tensor:
    return _evaluate_radial(
        q, k, v, plan, tensor_layout=tensor_layout, sm_scale=sm_scale
    )


def quantized_radial_attention_reference(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    value_fp16: torch.Tensor,
    plan: RadialAttentionPlan,
    *,
    tensor_layout: str = "HND",
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Oracle over the exact quantized operands consumed by the PPU kernel."""
    if tensor_layout == "NHD":
        q_len = query_int8.size(1)
        kv_len = key_int8.size(1)
        qh = query_int8.transpose(1, 2).float()
        kh = key_int8.transpose(1, 2).float()
    elif tensor_layout == "HND":
        q_len = query_int8.size(2)
        kv_len = key_int8.size(2)
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
    return _evaluate_radial(
        q_dequant,
        k_dequant,
        value_fp16,
        plan,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
    )


__all__ = [
    "quantized_radial_attention_reference",
    "radial_attention_reference",
]
