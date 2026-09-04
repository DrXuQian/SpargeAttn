"""Torch operator wrappers for the actlize-backed PPU SpargeAttention."""

import torch

from . import _qattn_ppu


def quant_per_warp_int8(q, k, km=None, tensor_layout="HND"):
    layout = 0 if tensor_layout == "NHD" else 1
    if tensor_layout == "HND":
        batch, q_heads, qo_len, _ = q.shape
        _, kv_heads, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        batch, qo_len, q_heads, _ = q.shape
        _, kv_len, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # Quant kernels promise contiguous Q/K to the attention kernel.  empty_like
    # may preserve a non-contiguous input memory format and silently violate
    # that contract.
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_blocks = ((qo_len + 127) // 128) * 4
    k_blocks = (kv_len + 63) // 64
    q_scale = torch.empty(
        (batch, q_heads, q_blocks), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (batch, kv_heads, k_blocks), device=k.device, dtype=torch.float32
    )
    _qattn_ppu.quant_per_warp_int8(q, q_int8, q_scale, 128, 32, layout)
    mean = torch.empty(0, device=k.device, dtype=k.dtype) if km is None else km
    _qattn_ppu.quant_per_block_int8(k, mean, k_int8, k_scale, 64, layout)
    return q_int8, q_scale, k_int8, k_scale


@torch.library.custom_op(
    "spas_sage_attn::qk_int8_sv_f16_block_sparse_accum_f32_attn_ppu",
    mutates_args=("output",),
    device_types="cuda",
)
def qk_int8_sv_f16_block_sparse_accum_f32_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_fp16: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    exact_row_ptr: torch.Tensor,
    exact_kv64: torch.Tensor,
    selected_route_bits: torch.Tensor,
    key_mean: torch.Tensor,
    value_mean: torch.Tensor,
    log2_block_counts: torch.Tensor,
    tensor_layout: int,
    query_block: int,
    route_block: int,
    use_summary: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _qattn_ppu.qk_int8_sv_f16_block_sparse_accum_f32_attn(
        query,
        key,
        value,
        query_fp16,
        output,
        query_scale,
        key_scale,
        exact_row_ptr,
        exact_kv64,
        selected_route_bits,
        key_mean,
        value_mean,
        log2_block_counts,
        tensor_layout,
        query_block,
        route_block,
        use_summary,
        sm_scale,
        return_lse,
    )


@torch.library.register_fake(
    "spas_sage_attn::qk_int8_sv_f16_block_sparse_accum_f32_attn_ppu"
)
def _qk_int8_sv_f16_block_sparse_accum_f32_attn_fake(
    query,
    key,
    value,
    query_fp16,
    output,
    query_scale,
    key_scale,
    exact_row_ptr,
    exact_kv64,
    selected_route_bits,
    key_mean,
    value_mean,
    log2_block_counts,
    tensor_layout,
    query_block,
    route_block,
    use_summary,
    sm_scale,
    return_lse,
):
    if return_lse:
        batch = query.size(0)
        if tensor_layout == 0:
            qo_len, heads = query.size(1), query.size(2)
        else:
            heads, qo_len = query.size(1), query.size(2)
        return torch.empty(
            (batch, heads, qo_len), device=query.device, dtype=torch.float32
        )
    return torch.empty((0,), device=query.device, dtype=torch.float32)


@torch.library.custom_op(
    "spas_sage_attn::qk_int8_sv_f16_radial_accum_f32_attn_ppu",
    mutates_args=("output",),
    device_types="cuda",
)
def qk_int8_sv_f16_radial_accum_f32_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    tensor_layout: int,
    sm_scale: float,
) -> torch.Tensor:
    return _qattn_ppu.qk_int8_sv_f16_radial_accum_f32_attn(
        query,
        key,
        value,
        output,
        query_scale,
        key_scale,
        block_lut,
        valid_block_num,
        tensor_layout,
        sm_scale,
    )


@torch.library.register_fake(
    "spas_sage_attn::qk_int8_sv_f16_radial_accum_f32_attn_ppu"
)
def _qk_int8_sv_f16_radial_accum_f32_attn_fake(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    block_lut,
    valid_block_num,
    tensor_layout,
    sm_scale,
):
    return torch.empty((0,), device=query.device, dtype=torch.float32)
