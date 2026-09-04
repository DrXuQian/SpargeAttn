/*
 * Copyright (c) 2026, SpargeAttn PPU contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * A single device executor for hard block-sparse, H3 top-k-summary and
 * Sol-summary plans.  Algorithm policy is resolved by the host planner; this
 * translation unit only consumes canonical KV64 CSR rows.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <torch/extension.h>

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_runtime.h>

#include "attn_ppu_ops.cuh"

namespace spargeattention::ppu::sparse {

constexpr int kBlockKV = 64;
constexpr int kWarpQ = 32;
constexpr float kLog2E = 1.4426950408889634f;

namespace detail {

template <int BlockQ, int HeadDim>
struct Accumulator {
  static constexpr int kQBlocksPerWarp = kWarpQ / 16;
  static constexpr int kVBlocks = HeadDim / 16;
  float out[kVBlocks][kQBlocksPerWarp][8];
  float row_max[kQBlocksPerWarp][2];
  float row_sum[kQBlocksPerWarp][2];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[d][qb][i] = 0.0f;
      }
    }
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        row_max[qb][row] = -1.0e30f;
        row_sum[qb][row] = 0.0f;
      }
    }
  }
};

template <int BlockQ, int HeadDim>
__device__ __forceinline__ void update_softmax(
    Accumulator<BlockQ, HeadDim> &acc,
    float (&score)[kWarpQ / 16][kBlockKV / 16][8]) {
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kVBlocks = HeadDim / 16;
#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int row_slot = 0; row_slot < 2; ++row_slot) {
      float tile_max = -1.0e30f;
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
        int const base = row_slot * 4;
        tile_max = fmaxf(tile_max,
            fmaxf(fmaxf(score[qb][kb][base], score[qb][kb][base + 1]),
                  fmaxf(score[qb][kb][base + 2], score[qb][kb][base + 3])));
      }
      tile_max = fmaxf(tile_max,
          __shfl_xor_sync(0xffffffffu, tile_max, 1));
      tile_max = fmaxf(tile_max,
          __shfl_xor_sync(0xffffffffu, tile_max, 2));
      float const new_max = fmaxf(tile_max, acc.row_max[qb][row_slot]);
      float const rescale = exp2f(acc.row_max[qb][row_slot] - new_max);
      acc.row_max[qb][row_slot] = new_max;
#pragma unroll
      for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
        for (int e = row_slot * 4; e < row_slot * 4 + 4; ++e) {
          acc.out[d][qb][e] *= rescale;
        }
      }
      float tile_sum = 0.0f;
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
        int const base = row_slot * 4;
        score[qb][kb][base] = exp2f(score[qb][kb][base] - new_max);
        score[qb][kb][base + 1] = exp2f(score[qb][kb][base + 1] - new_max);
        score[qb][kb][base + 2] = exp2f(score[qb][kb][base + 2] - new_max);
        score[qb][kb][base + 3] = exp2f(score[qb][kb][base + 3] - new_max);
        tile_sum += score[qb][kb][base] + score[qb][kb][base + 1]
            + score[qb][kb][base + 2] + score[qb][kb][base + 3];
      }
      tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
      tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
      acc.row_sum[qb][row_slot] =
          acc.row_sum[qb][row_slot] * rescale + tile_sum;
    }
  }
}

template <int BlockQ, int HeadDim>
__device__ __forceinline__ void multiply_probability_value(
    Accumulator<BlockQ, HeadDim> &acc,
    float (&score)[kWarpQ / 16][kBlockKV / 16][8],
    cutlass::half_t *smem_v) {
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kVBlocks = HeadDim / 16;
  constexpr int kHeadSlices = HeadDim / 64;
#pragma unroll
  for (int kb = 0; kb < kKVBlocks; ++kb) {
    uint32_t probability[kQBlocksPerWarp][4];
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
      score_accumulator_to_pv_operand(probability[qb], score[qb][kb]);
    }
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
      int const column = d * 16;
      uint32_t v_fragment[4];
      load_swizzled_transposed<kBlockKV, kHeadSlices>(
          v_fragment, smem_v, kb * 16, column % 64, column / 64);
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
        mma_f16f16f32(acc.out[d][qb], probability[qb], v_fragment);
      }
    }
  }
}

template <int BlockQ, int HeadDim>
__device__ __forceinline__ void score_exact_tile(
    float (&score)[kWarpQ / 16][kBlockKV / 16][8],
    int8_t *smem_q, int8_t *smem_k, int warp) {
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kSteps = HeadDim / 32;
  constexpr int kHeadSlices = HeadDim / 64;
#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) score[qb][kb][i] = 0.0f;
    }
  }
#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    uint32_t q_fragment[kQBlocksPerWarp][4];
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
      load_swizzled<int8_t, BlockQ, kHeadSlices>(
          q_fragment[qb], smem_q, warp * kWarpQ + qb * 16,
          (step & 1) * 32, step >> 1);
    }
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
      uint32_t k_fragment[4];
      load_swizzled<int8_t, kBlockKV, kHeadSlices>(
          k_fragment, smem_k, kb * 16,
          (step & 1) * 32, step >> 1);
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
        // score's storage is float so the same registers can feed the common
        // softmax path.  Use an explicit local integer accumulator per K step
        // only through a type-punned stable array below.
        auto *accum = reinterpret_cast<int32_t *>(score[qb][kb]);
        if (step == 0) {
#pragma unroll
          for (int i = 0; i < 8; ++i) accum[i] = 0;
        }
        mma_s8s8s32(*reinterpret_cast<int32_t (*)[8]>(accum),
                    q_fragment[qb], k_fragment);
      }
    }
  }
}

template <int BlockQ, int HeadDim>
__device__ __forceinline__ void score_summary_tile(
    float (&score)[kWarpQ / 16][kBlockKV / 16][8],
    cutlass::half_t *smem_q, cutlass::half_t *smem_k, int warp) {
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kSteps = HeadDim / 16;
  constexpr int kHeadSlices = HeadDim / 64;
#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) score[qb][kb][i] = 0.0f;
    }
  }
#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    uint32_t q_fragment[kQBlocksPerWarp][4];
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
      load_swizzled<cutlass::half_t, BlockQ, kHeadSlices>(
          q_fragment[qb], smem_q, warp * kWarpQ + qb * 16,
          (step & 3) * 16, step >> 2);
    }
#pragma unroll
    for (int kb = 0; kb < kKVBlocks; ++kb) {
      uint32_t k_fragment[4];
      load_swizzled<cutlass::half_t, kBlockKV, kHeadSlices>(
          k_fragment, smem_k, kb * 16,
          (step & 3) * 16, step >> 2);
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
        mma_f16f16f32(score[qb][kb], q_fragment[qb], k_fragment);
      }
    }
  }
}

template <int HeadDim, int BlockQ, bool UseSummary, bool ReturnLse,
          typename Output>
__launch_bounds__(BlockQ, 1)
__global__ void block_sparse_kernel(
    int8_t const *__restrict__ query,
    int8_t const *__restrict__ key,
    cutlass::half_t const *__restrict__ value,
    cutlass::half_t const *__restrict__ query_fp16,
    Output *__restrict__ output,
    float *__restrict__ lse,
    float const *__restrict__ query_scale,
    float const *__restrict__ key_scale,
    int32_t const *__restrict__ exact_row_ptr,
    int32_t const *__restrict__ exact_kv64,
    int32_t const *__restrict__ selected_route_bits,
    cutlass::half_t const *__restrict__ key_mean,
    cutlass::half_t const *__restrict__ value_mean,
    float const *__restrict__ log2_block_counts,
    int qo_len, int kv_len, int num_kv_heads, int route_blocks,
    int tensor_layout,
    float softmax_scale) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(BlockQ == 64 || BlockQ == 128);
  constexpr int kWarps = BlockQ / kWarpQ;
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kHeadSlices = HeadDim / 64;
  constexpr int kQScaleBlocksPer128 = 4;

  int const lane = int(threadIdx.x);
  int const warp = int(threadIdx.y);
  int const batch = int(blockIdx.z);
  int const head = int(blockIdx.y);
  int const num_qo_heads = int(gridDim.y);
  int const query_blocks = int(gridDim.x);
  int const route_words = (route_blocks + 31) / 32;
  // Reverse Q traversal exactly like the dense shipping kernel.  The plan row
  // remains the logical q_tile, never the physical launch ordinal.
  int const q_tile = int(gridDim.x) - 1 - int(blockIdx.x);
  int const q0 = q_tile * BlockQ;
  int const groups = num_qo_heads / num_kv_heads;
  int const kv_head = head / groups;
  int const plan_row = (batch * num_qo_heads + head) * query_blocks + q_tile;

  int const stride_seq_q = tensor_layout == 0 ? num_qo_heads * HeadDim : HeadDim;
  int const stride_h_q = tensor_layout == 0 ? HeadDim : qo_len * HeadDim;
  int const stride_seq_k = tensor_layout == 0 ? num_kv_heads * HeadDim : HeadDim;
  int const stride_h_k = tensor_layout == 0 ? HeadDim : kv_len * HeadDim;
  int const stride_bz_q = qo_len * num_qo_heads * HeadDim;
  int const stride_bz_k = kv_len * num_kv_heads * HeadDim;
  auto const *q_base = query + int64_t(batch) * stride_bz_q
      + int64_t(head) * stride_h_q;
  auto const *k_base = key + int64_t(batch) * stride_bz_k
      + int64_t(kv_head) * stride_h_k;
  auto const *v_base = value + int64_t(batch) * stride_bz_k
      + int64_t(kv_head) * stride_h_k;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto *smem_q = reinterpret_cast<int8_t *>(smem_raw);
  auto *smem_k = smem_q + BlockQ * HeadDim;
  auto *smem_v = reinterpret_cast<cutlass::half_t *>(
      smem_k + kBlockKV * HeadDim);

  auto issue_exact_k = [&](int kv_block) {
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
  auto issue_exact_v = [&](int kv_block) {
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

  if (lane == 0 && warp == 0) {
#pragma unroll
    for (int cube = 0; cube < kHeadSlices; ++cube) {
      aiu_load_swizzled_64<int8_t, BlockQ>(
          smem_q + cube * BlockQ * 64, q_base,
          qo_len, stride_seq_q, q0, cube * 64);
    }
  }
  commit_async_group();
  wait_async_group<0>();
  __syncthreads();

  Accumulator<BlockQ, HeadDim> accum;
  accum.clear();
  int const exact_begin = exact_row_ptr[plan_row];
  int const exact_end = exact_row_ptr[plan_row + 1];
  int const q_scale_blocks = ((qo_len + 127) / 128) * kQScaleBlocksPer128;
  float const q_scale = query_scale[
      (int64_t(batch) * num_qo_heads + head) * q_scale_blocks
      + (q0 + warp * kWarpQ) / kWarpQ];

  if (exact_begin < exact_end) issue_exact_k(exact_kv64[exact_begin]);
  for (int cursor = exact_begin; cursor < exact_end; ++cursor) {
    int const kv_block = exact_kv64[cursor];
    __syncthreads();
    issue_exact_v(kv_block);
    wait_async_group<1>();
    __syncthreads();

    float score[kQBlocksPerWarp][kKVBlocks][8];
    score_exact_tile<BlockQ, HeadDim>(score, smem_q, smem_k, warp);
    __syncthreads();
    issue_exact_k(cursor + 1 < exact_end ? exact_kv64[cursor + 1] : -1);

    float const score_scale = q_scale * key_scale[
        (int64_t(batch) * num_kv_heads + kv_head)
            * ((kv_len + kBlockKV - 1) / kBlockKV) + kv_block]
        * softmax_scale * kLog2E;
    bool const edge = (kv_block + 1) * kBlockKV > kv_len;
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int const kv_col = kv_block * kBlockKV + kb * 16
              + layout::accumulator_column(lane, e);
          auto *integer = reinterpret_cast<int32_t *>(score[qb][kb]);
          score[qb][kb][e] = edge && kv_col >= kv_len
              ? -1.0e30f : float(integer[e]) * score_scale;
        }
      }
    }
    update_softmax(accum, score);
    wait_async_group<1>();
    __syncthreads();
    multiply_probability_value(accum, score, smem_v);
  }

  if constexpr (UseSummary) {
    __syncthreads();
    auto *smem_qf = reinterpret_cast<cutlass::half_t *>(smem_raw);
    auto *smem_summary = smem_qf + BlockQ * HeadDim;
    auto const *qf_base = query_fp16 + int64_t(batch) * stride_bz_q
        + int64_t(head) * stride_h_q;
    auto const *km_base = key_mean
        + (int64_t(batch) * num_kv_heads + kv_head) * route_blocks * HeadDim;
    auto const *vm_base = value_mean
        + (int64_t(batch) * num_kv_heads + kv_head) * route_blocks * HeadDim;
    if (lane == 0 && warp == 0) {
#pragma unroll
      for (int cube = 0; cube < kHeadSlices; ++cube) {
        aiu_load_swizzled_64<cutlass::half_t, BlockQ>(
            smem_qf + cube * BlockQ * 64, qf_base,
            qo_len, stride_seq_q, q0, cube * 64);
      }
    }
    commit_async_group();
    wait_async_group<0>();
    __syncthreads();

    for (int summary0 = 0; summary0 < route_blocks; summary0 += kBlockKV) {
      if (lane == 0 && warp == 0) {
#pragma unroll
        for (int cube = 0; cube < kHeadSlices; ++cube) {
          aiu_load_swizzled_64<cutlass::half_t, kBlockKV>(
              smem_summary + cube * kBlockKV * 64, km_base,
              route_blocks, HeadDim, summary0, cube * 64);
        }
      }
      commit_async_group();
      wait_async_group<0>();
      __syncthreads();

      float score[kQBlocksPerWarp][kKVBlocks][8];
      score_summary_tile<BlockQ, HeadDim>(
          score, smem_qf, smem_summary, warp);
      float const score_scale = softmax_scale * kLog2E;
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
        for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            int const summary = summary0 + kb * 16
                + layout::accumulator_column(lane, e);
            bool selected = true;
            if (summary < route_blocks) {
              uint32_t const bits = static_cast<uint32_t>(
                  selected_route_bits[int64_t(plan_row) * route_words
                                      + summary / 32]);
              selected = ((bits >> (summary & 31)) & 1u) != 0;
            }
            score[qb][kb][e] = summary < route_blocks && !selected
                ? score[qb][kb][e] * score_scale
                    + log2_block_counts[summary]
                : -1.0e30f;
          }
        }
      }
      update_softmax(accum, score);
      __syncthreads();
      if (lane == 0 && warp == 0) {
#pragma unroll
        for (int cube = 0; cube < kHeadSlices; ++cube) {
          aiu_load_swizzled_64<cutlass::half_t, kBlockKV>(
              smem_summary + cube * kBlockKV * 64, vm_base,
              route_blocks, HeadDim, summary0, cube * 64);
        }
      }
      commit_async_group();
      wait_async_group<0>();
      __syncthreads();
      multiply_probability_value(accum, score, smem_summary);
      __syncthreads();
    }
  }

  constexpr int kVBlocks = HeadDim / 16;
#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int const q_in_tile = warp * kWarpQ + qb * 16
            + layout::accumulator_row(lane, e);
        int const q_row = q0 + q_in_tile;
        if (q_row < qo_len) {
          int const column = d * 16 + layout::accumulator_column(lane, e);
          float const denominator =
              accum.row_sum[qb][layout::accumulator_row_slot(e)];
          output[int64_t(batch) * stride_bz_q + int64_t(q_row) * stride_seq_q
                 + int64_t(head) * stride_h_q + column] =
              convert_output<Output>(accum.out[d][qb][e] / denominator);
        }
      }
    }
  }
  if constexpr (ReturnLse) {
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int row_slot = 0; row_slot < 2; ++row_slot) {
        int const q_in_tile = warp * kWarpQ + qb * 16
            + layout::accumulator_row(lane, row_slot * 4);
        int const q_row = q0 + q_in_tile;
        if ((lane & 3) == 0 && q_row < qo_len) {
          lse[(int64_t(batch) * num_qo_heads + head) * qo_len + q_row] =
              accum.row_max[qb][row_slot]
              + log2f(accum.row_sum[qb][row_slot]);
        }
      }
    }
  }
}

template <int HeadDim, int BlockQ, bool UseSummary, bool ReturnLse,
          typename Output>
void launch(
    torch::Tensor const &query, torch::Tensor const &key,
    torch::Tensor const &value, torch::Tensor const &query_fp16,
    torch::Tensor const &output, torch::Tensor const &lse,
    torch::Tensor const &query_scale, torch::Tensor const &key_scale,
    torch::Tensor const &row_ptr, torch::Tensor const &indices,
    torch::Tensor const &selected_bits, torch::Tensor const &key_mean,
    torch::Tensor const &value_mean, torch::Tensor const &log2_counts,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int query_blocks, int route_blocks, int tensor_layout,
    float softmax_scale) {
  constexpr size_t exact_smem =
      BlockQ * HeadDim * sizeof(int8_t)
      + kBlockKV * HeadDim * sizeof(int8_t)
      + kBlockKV * HeadDim * sizeof(cutlass::half_t);
  constexpr size_t summary_smem =
      BlockQ * HeadDim * sizeof(cutlass::half_t)
      + kBlockKV * HeadDim * sizeof(cutlass::half_t);
  constexpr size_t smem_bytes = UseSummary
      ? (exact_smem > summary_smem ? exact_smem : summary_smem)
      : exact_smem;
  auto kernel = &block_sparse_kernel<
      HeadDim, BlockQ, UseSummary, ReturnLse, Output>;
  static hggcError_t const attribute_status = hggcFuncSetAttribute(
      reinterpret_cast<void const *>(kernel),
      hggcFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
  TORCH_CHECK(attribute_status == hggcSuccess,
              "PPU SpargeAttention dynamic-smem opt-in failed: ",
              hggcGetErrorString(attribute_status));
  dim3 const grid(query_blocks, num_qo_heads, query.size(0));
  dim3 const block(32, BlockQ / kWarpQ);
  kernel<<<grid, block, smem_bytes>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(),
      reinterpret_cast<cutlass::half_t const *>(value.data_ptr()),
      UseSummary
          ? reinterpret_cast<cutlass::half_t const *>(query_fp16.data_ptr())
          : nullptr,
      reinterpret_cast<Output *>(output.data_ptr()),
      ReturnLse ? lse.data_ptr<float>() : nullptr,
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
      row_ptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
      selected_bits.data_ptr<int32_t>(),
      reinterpret_cast<cutlass::half_t const *>(key_mean.data_ptr()),
      reinterpret_cast<cutlass::half_t const *>(value_mean.data_ptr()),
      log2_counts.data_ptr<float>(),
      qo_len, kv_len, num_kv_heads, route_blocks, tensor_layout,
      softmax_scale);
  auto const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess,
              "PPU SpargeAttention launch failed: ",
              hggcGetErrorString(error));
}

}  // namespace detail
}  // namespace spargeattention::ppu::sparse

torch::Tensor qk_int8_sv_f16_block_sparse_accum_f32_attn_ppu(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor query_fp16, torch::Tensor output,
    torch::Tensor query_scale, torch::Tensor key_scale,
    torch::Tensor exact_row_ptr, torch::Tensor exact_kv64,
    torch::Tensor selected_route_bits, torch::Tensor key_mean,
    torch::Tensor value_mean, torch::Tensor log2_block_counts,
    int tensor_layout, int query_block, int route_block,
    int use_summary, float softmax_scale, int return_lse) {
  using namespace spargeattention::ppu::sparse;
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1,
              "tensor_layout must be 0 (NHD) or 1 (HND)");
  TORCH_CHECK(query_block == 64 || query_block == 128,
              "PPU sparse query_block must be 64 or 128");
  TORCH_CHECK(route_block == 64 || route_block == 128,
              "PPU sparse route_block must be 64 or 128");
  TORCH_CHECK(route_block % kBlockKV == 0,
              "PPU sparse route_block must be divisible by KV64");
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda()
                  && output.is_cuda() && query_scale.is_cuda()
                  && key_scale.is_cuda() && exact_row_ptr.is_cuda()
                  && exact_kv64.is_cuda() && selected_route_bits.is_cuda()
                  && key_mean.is_cuda() && value_mean.is_cuda()
                  && log2_block_counts.is_cuda(),
              "PPU SpargeAttention tensors must be device tensors");
  if (use_summary) {
    TORCH_CHECK(query_fp16.is_cuda(),
                "PPU summary attention requires a device fp16 Q tensor");
  }
  int const device = query.get_device();
  for (auto const &tensor : {key, value, output, query_scale, key_scale,
                             exact_row_ptr, exact_kv64, selected_route_bits,
                             key_mean, value_mean, log2_block_counts}) {
    TORCH_CHECK(tensor.get_device() == device,
                "all PPU sparse tensors must share one device");
  }
  TORCH_CHECK(query.scalar_type() == torch::kInt8
                  && key.scalar_type() == torch::kInt8,
              "PPU sparse Q/K must be int8");
  TORCH_CHECK(value.scalar_type() == torch::kHalf
                  && (!use_summary || query_fp16.scalar_type() == torch::kHalf)
                  && key_mean.scalar_type() == torch::kHalf
                  && value_mean.scalar_type() == torch::kHalf,
              "PPU sparse V/Q-summary/means must be fp16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "the first PPU sparse shipping path requires bf16 output");
  TORCH_CHECK(query_scale.scalar_type() == torch::kFloat32
                  && key_scale.scalar_type() == torch::kFloat32
                  && log2_block_counts.scalar_type() == torch::kFloat32,
              "PPU sparse scales/counts must be fp32");
  TORCH_CHECK(exact_row_ptr.scalar_type() == torch::kInt32
                  && exact_kv64.scalar_type() == torch::kInt32
                  && selected_route_bits.scalar_type() == torch::kInt32,
              "PPU sparse plan indices and bits must be int32");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4
                  && output.dim() == 4 && query_scale.dim() == 3
                  && key_scale.dim() == 3 && exact_row_ptr.dim() == 1
                  && exact_kv64.dim() == 1 && selected_route_bits.dim() == 2
                  && key_mean.dim() == 4 && value_mean.dim() == 4
                  && log2_block_counts.dim() == 1,
              "PPU sparse tensor ranks do not match the ABI");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous()
                  && value.is_contiguous() && output.is_contiguous()
                  && (!use_summary || query_fp16.is_contiguous())
                  && query_scale.is_contiguous() && key_scale.is_contiguous()
                  && exact_row_ptr.is_contiguous() && exact_kv64.is_contiguous()
                  && selected_route_bits.is_contiguous()
                  && key_mean.is_contiguous() && value_mean.is_contiguous()
                  && log2_block_counts.is_contiguous(),
              "PPU sparse plan/Q/K/scale tensors must be contiguous");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "PPU sparse output shape must equal query shape");

  int const batch = int(query.size(0));
  int const head_dim = int(query.size(3));
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
  TORCH_CHECK(head_dim == 128,
              "the first PPU sparse shipping path requires head_dim 128");
  TORCH_CHECK(!return_lse,
              "the first PPU sparse shipping path does not publish LSE");
  TORCH_CHECK(qo_len > 0 && kv_len > 0 && num_qo_heads > 0
                  && num_kv_heads > 0 && num_qo_heads % num_kv_heads == 0,
              "PPU sparse dimensions/head grouping are invalid");
  int const query_blocks = (qo_len + query_block - 1) / query_block;
  int const route_blocks = (kv_len + route_block - 1) / route_block;
  int const route_words = (route_blocks + 31) / 32;
  int64_t const rows = int64_t(batch) * num_qo_heads * query_blocks;
  TORCH_CHECK(exact_row_ptr.numel() == rows + 1,
              "PPU sparse row_ptr length mismatch");
  TORCH_CHECK(selected_route_bits.size(0) == rows
                  && selected_route_bits.size(1) == route_words,
              "PPU sparse selected-route bitset shape mismatch");
  TORCH_CHECK(key_mean.size(0) == batch && value_mean.sizes() == key_mean.sizes()
                  && key_mean.size(1) == num_kv_heads
                  && key_mean.size(2) == route_blocks
                  && key_mean.size(3) == head_dim,
              "PPU sparse summary shape mismatch");
  TORCH_CHECK(log2_block_counts.numel() == route_blocks,
              "PPU sparse summary count mismatch");
  TORCH_CHECK(query_scale.size(0) == batch
                  && query_scale.size(1) == num_qo_heads
                  && query_scale.size(2) == ((qo_len + 127) / 128) * 4,
              "PPU sparse Q scale shape mismatch");
  TORCH_CHECK(key_scale.size(0) == batch && key_scale.size(1) == num_kv_heads
                  && key_scale.size(2) == (kv_len + 63) / 64,
              "PPU sparse K scale shape mismatch");

  torch::Tensor lse = return_lse
      ? torch::empty({batch, num_qo_heads, qo_len},
                     query.options().dtype(torch::kFloat32))
      : torch::empty({0}, query.options().dtype(torch::kFloat32));

#define PPU_SPARSE_LAUNCH(HD, QB, SUMMARY, LSE, OUT)                         \
  spargeattention::ppu::sparse::detail::launch<HD, QB, SUMMARY, LSE, OUT>( \
      query, key, value, query_fp16, output, lse, query_scale, key_scale,   \
      exact_row_ptr, exact_kv64, selected_route_bits, key_mean, value_mean, \
      log2_block_counts, qo_len, kv_len, num_qo_heads, num_kv_heads,        \
      query_blocks, route_blocks, tensor_layout, softmax_scale)
#define PPU_SPARSE_FLAGS(HD, QB, OUT)                                       \
  do {                                                                       \
    if (use_summary) {                                                       \
      PPU_SPARSE_LAUNCH(HD, QB, true, false, OUT);                           \
    } else {                                                                 \
      PPU_SPARSE_LAUNCH(HD, QB, false, false, OUT);                          \
    }                                                                        \
  } while (false)
  if (query_block == 64) {
    PPU_SPARSE_FLAGS(128, 64, cutlass::bfloat16_t);
  } else {
    PPU_SPARSE_FLAGS(128, 128, cutlass::bfloat16_t);
  }
#undef PPU_SPARSE_FLAGS
#undef PPU_SPARSE_LAUNCH
  return lse;
}
