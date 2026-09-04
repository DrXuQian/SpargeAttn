/* Copyright (c) 2026, SpargeAttn PPU contributors. */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "attn_ppu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "qk_int8_sv_f16_block_sparse_accum_f32_attn",
      &qk_int8_sv_f16_block_sparse_accum_f32_attn_ppu,
      "PPU block-sparse QK-int8 / PV-fp16 SpargeAttention");
  module.def(
      "qk_int8_sv_f16_radial_accum_f32_attn",
      &qk_int8_sv_f16_radial_accum_f32_attn_ppu,
      "PPU Radial/SparseSage2 QK-int8 / PV-fp16 SpargeAttention");
  module.def(
      "quant_per_warp_int8",
      &quant_per_warp_int8_ppu,
      "PPU per-warp INT8 quantization for Q");
  module.def(
      "quant_per_block_int8",
      &quant_per_block_int8_ppu,
      "PPU per-block INT8 quantization for K");
}
