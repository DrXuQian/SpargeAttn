/*
 * Copyright (c) 2026, SpargeAttn PPU contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * PPU port of the execution contract used by RadialAttention's
 * SparseSageAttention2 backend.  Mask policy stays on the host.  The kernel
 * consumes its Q128/KV64 delta-coded LUT directly; it is intentionally not a
 * compatibility branch in the generic H3/Sol CSR+summary executor.
 */

#include <algorithm>
#include <cstdint>

#include <torch/extension.h>

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_runtime.h>

#include "attn_ppu_ops.cuh"

namespace spargeattention::ppu::radial {

constexpr int kHeadDim = 128;
constexpr int kBlockQ = 128;
constexpr int kBlockKV = 64;
// SparseSageAttention2's FP16-V/D128 path assigns one 16-row Q tile per warp.
constexpr int kWarpQ = 16;
constexpr int kWarps = kBlockQ / kWarpQ;
constexpr int kKVBlocks = kBlockKV / 16;
constexpr int kVBlocks = kHeadDim / 16;
constexpr int kHeadSlices = kHeadDim / 64;
constexpr float kLog2E = 1.4426950408889634f;

struct Accumulator {
  float out[kVBlocks][8];
  float row_max[2];
  float row_sum[2];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[d][i] = 0.0f;
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      row_max[row] = -1.0e30f;
      row_sum[row] = 0.0f;
    }
  }
};

union ScoreStorage {
  int32_t integer[kKVBlocks][8];
  float fp32[kKVBlocks][8];
};

__device__ __forceinline__ void score_qk(
    ScoreStorage &score, int8_t *smem_q, int8_t *smem_k, int warp) {
#pragma unroll
  for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) score.integer[kb][i] = 0;
  }
#pragma unroll
  for (int step = 0; step < kHeadDim / 32; ++step) {
    uint32_t q_fragment[4];
    load_swizzled<int8_t, kBlockQ, kHeadSlices>(
        q_fragment, smem_q, warp * kWarpQ,
        (step & 1) * 32, step >> 1);
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
      uint32_t k_fragment[4];
      load_swizzled<int8_t, kBlockKV, kHeadSlices>(
          k_fragment, smem_k, kb * 16,
          (step & 1) * 32, step >> 1);
      mma_s8s8s32(score.integer[kb], q_fragment, k_fragment);
    }
  }
}

__device__ __forceinline__ void update_softmax(
    Accumulator &acc, ScoreStorage &score) {
#pragma unroll
  for (int row_slot = 0; row_slot < 2; ++row_slot) {
    int const base = row_slot * 4;
    float tile_max = -1.0e30f;
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
      float const *s = score.fp32[kb];
      tile_max = fmaxf(
          tile_max,
          fmaxf(fmaxf(s[base], s[base + 1]),
                fmaxf(s[base + 2], s[base + 3])));
    }
    tile_max = fmaxf(tile_max,
                     __shfl_xor_sync(0xffffffffu, tile_max, 1));
    tile_max = fmaxf(tile_max,
                     __shfl_xor_sync(0xffffffffu, tile_max, 2));
    float const new_max = fmaxf(tile_max, acc.row_max[row_slot]);
    float const rescale = exp2f(acc.row_max[row_slot] - new_max);
    acc.row_max[row_slot] = new_max;
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
      for (int e = base; e < base + 4; ++e) acc.out[d][e] *= rescale;
    }
    float tile_sum = 0.0f;
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
      float *s = score.fp32[kb];
      s[base] = exp2f(s[base] - new_max);
      s[base + 1] = exp2f(s[base + 1] - new_max);
      s[base + 2] = exp2f(s[base + 2] - new_max);
      s[base + 3] = exp2f(s[base + 3] - new_max);
      tile_sum += s[base] + s[base + 1] + s[base + 2] + s[base + 3];
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    acc.row_sum[row_slot] = acc.row_sum[row_slot] * rescale + tile_sum;
  }
}

__device__ __forceinline__ void multiply_pv(
    Accumulator &acc, ScoreStorage &score, cutlass::half_t *smem_v) {
#pragma unroll
  for (int kb = 0; kb < kKVBlocks; ++kb) {
    uint32_t probability[4];
    score_accumulator_to_pv_operand(probability, score.fp32[kb]);
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
      int const column = d * 16;
      uint32_t v_fragment[4];
      load_swizzled_transposed<kBlockKV, kHeadSlices>(
          v_fragment, smem_v, kb * 16, column % 64, column / 64);
      mma_f16f16f32(acc.out[d], probability, v_fragment);
    }
  }
}

__launch_bounds__(256, 1)
__global__ void radial_sparse_kernel(
    int8_t const *__restrict__ query,
    int8_t const *__restrict__ key,
    cutlass::half_t const *__restrict__ value,
    cutlass::bfloat16_t *__restrict__ output,
    float const *__restrict__ query_scale,
    float const *__restrict__ key_scale,
    int32_t const *__restrict__ block_lut,
    int32_t const *__restrict__ valid_block_num,
    int qo_len, int kv_len, int num_kv_heads, int tensor_layout,
    float softmax_scale) {
  int const lane = int(threadIdx.x);
  int const warp = int(threadIdx.y);
  int const q_tile = int(blockIdx.x);
  int const head = int(blockIdx.y);
  int const batch = int(blockIdx.z);
  int const num_qo_heads = int(gridDim.y);
  int const query_blocks = int(gridDim.x);
  int const kv_blocks = (kv_len + kBlockKV - 1) / kBlockKV;
  int const q0 = q_tile * kBlockQ;
  int const groups = num_qo_heads / num_kv_heads;
  int const kv_head = head / groups;
  int64_t const plan_row =
      (int64_t(batch) * num_qo_heads + head) * query_blocks + q_tile;
  int32_t const *lut = block_lut + plan_row * kv_blocks;
  int const iterations = valid_block_num[plan_row];
  if (iterations <= 0) return;

  int const stride_seq_q = tensor_layout == 0 ? num_qo_heads * kHeadDim
                                               : kHeadDim;
  int const stride_h_q = tensor_layout == 0 ? kHeadDim
                                             : qo_len * kHeadDim;
  int const stride_seq_k = tensor_layout == 0 ? num_kv_heads * kHeadDim
                                               : kHeadDim;
  int const stride_h_k = tensor_layout == 0 ? kHeadDim
                                             : kv_len * kHeadDim;
  int64_t const stride_bz_q = int64_t(qo_len) * num_qo_heads * kHeadDim;
  int64_t const stride_bz_k = int64_t(kv_len) * num_kv_heads * kHeadDim;
  auto const *q_base = query + int64_t(batch) * stride_bz_q
      + int64_t(head) * stride_h_q;
  auto const *k_base = key + int64_t(batch) * stride_bz_k
      + int64_t(kv_head) * stride_h_k;
  auto const *v_base = value + int64_t(batch) * stride_bz_k
      + int64_t(kv_head) * stride_h_k;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto *smem_q = reinterpret_cast<int8_t *>(smem_raw);
  auto *smem_k = smem_q + kBlockQ * kHeadDim;
  auto *smem_v = reinterpret_cast<cutlass::half_t *>(
      smem_k + kBlockKV * kHeadDim);

  auto issue_k = [&](int kv_block) {
    if (kv_block >= 0 && lane == 0 && warp == 0) {
#pragma unroll
      for (int cube = 0; cube < kHeadSlices; ++cube) {
        aiu_load_swizzled_64<int8_t, kBlockKV>(
            smem_k + cube * kBlockKV * 64, k_base,
            kv_len, stride_seq_k, kv_block * kBlockKV, cube * 64);
      }
    }
    commit_async_group();
  };
  auto issue_v = [&](int kv_block) {
    if (lane == 0 && warp == 0) {
#pragma unroll
      for (int cube = 0; cube < kHeadSlices; ++cube) {
        aiu_load_swizzled_64<cutlass::half_t, kBlockKV>(
            smem_v + cube * kBlockKV * 64, v_base,
            kv_len, stride_seq_k, kv_block * kBlockKV, cube * 64);
      }
    }
    commit_async_group();
  };

  // Q is invariant across the entire LUT row and is loaded exactly once.
  if (lane == 0 && warp == 0) {
#pragma unroll
    for (int cube = 0; cube < kHeadSlices; ++cube) {
      aiu_load_swizzled_64<int8_t, kBlockQ>(
          smem_q + cube * kBlockQ * 64, q_base,
          qo_len, stride_seq_q, q0, cube * 64);
    }
  }
  commit_async_group();
  wait_async_group<0>();
  __syncthreads();

  Accumulator accum;
  accum.clear();
  // Q quantization is per 32 rows.  Two neighboring 16-row compute warps
  // deliberately consume the same measured scale.
  float const q_scale = query_scale[
      (int64_t(batch) * num_qo_heads + head) * query_blocks * 4
      + q_tile * 4 + warp / 2];

  int kv_block = lut[0];
  issue_k(kv_block);
  for (int iter = 0; iter < iterations; ++iter) {
    __syncthreads();
    issue_v(kv_block);
    // K is the older of the two pending groups; V remains in flight.
    wait_async_group<1>();
    __syncthreads();

    ScoreStorage score;
    score_qk(score, smem_q, smem_k, warp);

    __syncthreads();
    int next_kv_block = -1;
    if (iter + 1 < iterations) {
      next_kv_block = kv_block + lut[iter + 1];
    }
    issue_k(next_kv_block);

    float const scale = q_scale * key_scale[
        (int64_t(batch) * num_kv_heads + kv_head) * kv_blocks + kv_block]
        * softmax_scale * kLog2E;
    bool const edge = (kv_block + 1) * kBlockKV > kv_len;
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int const kv_col = kv_block * kBlockKV + kb * 16
            + layout::accumulator_column(lane, e);
        score.fp32[kb][e] = edge && kv_col >= kv_len
            ? -1.0e30f : float(score.integer[kb][e]) * scale;
      }
    }
    update_softmax(accum, score);

    // V is now the oldest pending group; next K may continue in flight.
    wait_async_group<1>();
    __syncthreads();
    multiply_pv(accum, score, smem_v);
    kv_block = next_kv_block;
  }

#pragma unroll
  for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int const q_in_tile = warp * kWarpQ
          + layout::accumulator_row(lane, e);
      int const q_row = q0 + q_in_tile;
      if (q_row < qo_len) {
        int const column = d * 16 + layout::accumulator_column(lane, e);
        float const denominator =
            accum.row_sum[layout::accumulator_row_slot(e)];
        output[int64_t(batch) * stride_bz_q + int64_t(q_row) * stride_seq_q
               + int64_t(head) * stride_h_q + column] =
            cutlass::bfloat16_t(accum.out[d][e] / denominator);
      }
    }
  }
}

void launch(
    torch::Tensor const &query, torch::Tensor const &key,
    torch::Tensor const &value, torch::Tensor const &output,
    torch::Tensor const &query_scale, torch::Tensor const &key_scale,
    torch::Tensor const &block_lut, torch::Tensor const &valid_block_num,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int tensor_layout, float softmax_scale) {
  constexpr size_t smem_bytes =
      kBlockQ * kHeadDim * sizeof(int8_t)
      + kBlockKV * kHeadDim * sizeof(int8_t)
      + kBlockKV * kHeadDim * sizeof(cutlass::half_t);
  auto kernel = &radial_sparse_kernel;
  static hggcError_t const attribute_status = hggcFuncSetAttribute(
      reinterpret_cast<void const *>(kernel),
      hggcFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
  TORCH_CHECK(attribute_status == hggcSuccess,
              "PPU Radial dynamic-smem opt-in failed: ",
              hggcGetErrorString(attribute_status));
  dim3 const grid((qo_len + kBlockQ - 1) / kBlockQ,
                  num_qo_heads, query.size(0));
  dim3 const block(32, kWarps);
  kernel<<<grid, block, smem_bytes>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(),
      reinterpret_cast<cutlass::half_t const *>(value.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t *>(output.data_ptr()),
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
      block_lut.data_ptr<int32_t>(), valid_block_num.data_ptr<int32_t>(),
      qo_len, kv_len, num_kv_heads, tensor_layout, softmax_scale);
  auto const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess,
              "PPU Radial launch failed: ", hggcGetErrorString(error));
}

}  // namespace spargeattention::ppu::radial

torch::Tensor qk_int8_sv_f16_radial_accum_f32_attn_ppu(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor output, torch::Tensor query_scale,
    torch::Tensor key_scale, torch::Tensor block_lut,
    torch::Tensor valid_block_num, int tensor_layout,
    float softmax_scale) {
  using namespace spargeattention::ppu::radial;
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1,
              "tensor_layout must be 0 (NHD) or 1 (HND)");
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda()
                  && output.is_cuda() && query_scale.is_cuda()
                  && key_scale.is_cuda() && block_lut.is_cuda()
                  && valid_block_num.is_cuda(),
              "PPU Radial tensors must be device tensors");
  int const device = query.get_device();
  for (auto const &tensor : {key, value, output, query_scale, key_scale,
                             block_lut, valid_block_num}) {
    TORCH_CHECK(tensor.get_device() == device,
                "all PPU Radial tensors must share one device");
  }
  TORCH_CHECK(query.scalar_type() == torch::kInt8
                  && key.scalar_type() == torch::kInt8,
              "PPU Radial Q/K must be int8");
  TORCH_CHECK(value.scalar_type() == torch::kHalf,
              "the first PPU Radial path requires fp16 V");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "the first PPU Radial path requires bf16 output");
  TORCH_CHECK(query_scale.scalar_type() == torch::kFloat32
                  && key_scale.scalar_type() == torch::kFloat32,
              "PPU Radial Q/K scales must be fp32");
  TORCH_CHECK(block_lut.scalar_type() == torch::kInt32
                  && valid_block_num.scalar_type() == torch::kInt32,
              "PPU Radial LUT/counts must be int32");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4
                  && output.dim() == 4 && query_scale.dim() == 3
                  && key_scale.dim() == 3 && block_lut.dim() == 4
                  && valid_block_num.dim() == 3,
              "PPU Radial tensor ranks do not match the ABI");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous()
                  && value.is_contiguous() && output.is_contiguous()
                  && query_scale.is_contiguous() && key_scale.is_contiguous()
                  && block_lut.is_contiguous()
                  && valid_block_num.is_contiguous(),
              "PPU Radial tensors must be contiguous");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "PPU Radial output shape must equal query shape");

  int const batch = int(query.size(0));
  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  if (tensor_layout == 0) {
    qo_len = int(query.size(1));
    kv_len = int(key.size(1));
    num_qo_heads = int(query.size(2));
    num_kv_heads = int(key.size(2));
  } else {
    qo_len = int(query.size(2));
    kv_len = int(key.size(2));
    num_qo_heads = int(query.size(1));
    num_kv_heads = int(key.size(1));
  }
  TORCH_CHECK(query.size(3) == kHeadDim && key.size(3) == kHeadDim
                  && value.size(3) == kHeadDim,
              "the first PPU Radial path requires head_dim 128");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "PPU Radial K/V shapes must match");
  TORCH_CHECK(batch > 0 && qo_len > 0 && kv_len > 0
                  && num_qo_heads > 0 && num_kv_heads > 0
                  && num_qo_heads % num_kv_heads == 0,
              "PPU Radial dimensions/head grouping are invalid");
  int const q_blocks = (qo_len + kBlockQ - 1) / kBlockQ;
  int const kv_blocks = (kv_len + kBlockKV - 1) / kBlockKV;
  TORCH_CHECK(query_scale.size(0) == batch
                  && query_scale.size(1) == num_qo_heads
                  && query_scale.size(2) == q_blocks * 4,
              "PPU Radial Q scale shape mismatch");
  TORCH_CHECK(key_scale.size(0) == batch
                  && key_scale.size(1) == num_kv_heads
                  && key_scale.size(2) == kv_blocks,
              "PPU Radial K scale shape mismatch");
  TORCH_CHECK(block_lut.size(0) == batch
                  && block_lut.size(1) == num_qo_heads
                  && block_lut.size(2) == q_blocks
                  && block_lut.size(3) == kv_blocks,
              "PPU Radial LUT shape mismatch");
  TORCH_CHECK(valid_block_num.size(0) == batch
                  && valid_block_num.size(1) == num_qo_heads
                  && valid_block_num.size(2) == q_blocks,
              "PPU Radial valid-count shape mismatch");

  launch(query, key, value, output, query_scale, key_scale,
         block_lut, valid_block_num, qo_len, kv_len,
         num_qo_heads, num_kv_heads, tensor_layout, softmax_scale);
  return torch::empty({0}, query.options().dtype(torch::kFloat32));
}
