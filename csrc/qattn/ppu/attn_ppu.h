/* Copyright (c) 2026, SpargeAttn PPU contributors. */
#pragma once

#include <torch/extension.h>

torch::Tensor qk_int8_sv_f16_block_sparse_accum_f32_attn_ppu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_fp16,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor exact_row_ptr,
    torch::Tensor exact_kv64,
    torch::Tensor selected_route_bits,
    torch::Tensor key_mean,
    torch::Tensor value_mean,
    torch::Tensor log2_block_counts,
    int tensor_layout,
    int query_block,
    int route_block,
    int use_summary,
    float softmax_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f16_radial_accum_f32_attn_ppu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor block_lut,
    torch::Tensor valid_block_num,
    int tensor_layout,
    float softmax_scale);

void quant_per_warp_int8_ppu(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int warp_block_size,
    int tensor_layout);

void quant_per_block_int8_ppu(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int tensor_layout);
