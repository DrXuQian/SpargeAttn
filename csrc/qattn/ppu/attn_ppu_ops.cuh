/*
 * Copyright (c) 2026, SpargeAttn PPU contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <cstdint>

#include <hggc_fp16.h>

#include <cute/arch/copy_ppu.hpp>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cutlass/numeric_types.h>

#include "attn_ppu_layout.cuh"

namespace spargeattention::ppu {

template <typename Element, int CubeH>
__device__ __forceinline__ void aiu_load_swizzled_64(
    Element *smem, Element const *gmem, int dim_h, int dim_w,
    int coord_h, int coord_w) {
  constexpr int CubeW = 64;
  cute::AiuDesc desc{};
  desc.gmem_ptr = reinterpret_cast<uint8_t const *>(gmem);
  desc.dim_h = dim_h;
  desc.dim_w = dim_w;
  desc.cube_h = CubeH;
  desc.cube_w = CubeW;
  desc.offset_w = 0;
  using Copy = cute::PPU0010_AIU_LOAD<
      cute::C<CubeH * CubeW * int(sizeof(Element)) * 8>,
      Element, false, true>;
  Copy::copy(smem, gmem, desc, coord_w, coord_h);
}

template <typename Element, int CubeH, int Cubes>
__device__ __forceinline__ void load_swizzled(
    uint32_t (&fragment)[4], Element *smem, int coord_h, int coord_w,
    int cube) {
  using Copy = cute::PPU0010_TSM_LD_SWZL<
      Element, CubeH, 64, false, false, Cubes>;
  Copy::copy(fragment, smem, coord_w, coord_h, cube);
}

template <int CubeH, int Cubes>
__device__ __forceinline__ void load_swizzled_transposed(
    uint32_t (&fragment)[4], cutlass::half_t *smem,
    int coord_h, int coord_w, int cube) {
  using Copy = cute::PPU0010_TSM_LD_SWZL<
      cutlass::half_t, CubeH, 64, false, true, Cubes>;
  Copy::copy(fragment, smem, coord_h, coord_w, cube);
}

__device__ __forceinline__ void mma_s8s8s32(
    int32_t (&d)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  auto *du = reinterpret_cast<uint32_t *>(d);
  cute::PPU0010_16x16x32_S32S8S8S32_TN::fma(
      du[0], du[1], du[2], du[3], du[4], du[5], du[6], du[7],
      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
      du[0], du[1], du[2], du[3], du[4], du[5], du[6], du[7]);
}

__device__ __forceinline__ void mma_f16f16f32(
    float (&d)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  cute::PPU0010_16x16x16_F32F16F16F32_TN::fma(
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

// CLayout and ALayout are different physical views on PPU0010.  Put the score
// in operand A and the per-lane permutation matrix in B.  The L221 host oracle
// derives the bridge from the real traits (M is the fastest logical mode) and
// proves all 256 values.  The same physical operation is independently
// device-anchored by fattn_ppu.cu; swapping A/B is 240/256 wrong.
__device__ __forceinline__ void score_accumulator_to_pv_operand(
    uint32_t (&dst)[4], float const (&score)[8]) {
  uint32_t a[4];
  auto *packed = reinterpret_cast<__half2 *>(a);
  packed[0] = __floats2half2_rn(score[0], score[1]);
  packed[1] = __floats2half2_rn(score[2], score[3]);
  packed[2] = __floats2half2_rn(score[4], score[5]);
  packed[3] = __floats2half2_rn(score[6], score[7]);

  int const lane = int(threadIdx.x) & 31;
  uint32_t const b[4] = {
      layout::pv_bridge_b_word(lane, 0),
      layout::pv_bridge_b_word(lane, 1),
      layout::pv_bridge_b_word(lane, 2),
      layout::pv_bridge_b_word(lane, 3)};
  uint32_t const zero = 0;

  cute::PPU0010_16x16x16_F16F16F16F16_TN::fma(
      dst[0], dst[1], dst[2], dst[3],
      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
      zero, zero, zero, zero);
}

__device__ __forceinline__ void commit_async_group() {
  cute::cp_async_fence();
}

template <int Pending>
__device__ __forceinline__ void wait_async_group() {
  cute::cp_async_wait<Pending>();
}

template <typename Output>
__device__ __forceinline__ Output convert_output(float value) {
  return Output(value);
}

}  // namespace spargeattention::ppu
