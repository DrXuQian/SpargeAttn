/* Copyright (c) 2026, SpargeAttn PPU contributors. */

#include <cstdint>

#include <torch/extension.h>

#include <hggc_runtime.h>

#include <cutlass/numeric_types.h>

namespace spargeattention::ppu {
namespace detail {

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return float(value);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
  }
  return value;
}

__device__ __forceinline__ float block_max(float value) {
  __shared__ float warp_values[32];
  int const lane = int(threadIdx.x) & 31;
  int const warp = int(threadIdx.x) >> 5;
  value = warp_max(value);
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  value = lane < (int(blockDim.x) + 31) / 32 ? warp_values[lane] : 0.0f;
  return warp_max(value);
}

template <typename Input, int HeadDim, int Rows, bool SubtractMean>
__global__ void quantize_int8_kernel(
    Input const *__restrict__ input,
    Input const *__restrict__ mean,
    int8_t *__restrict__ output,
    float *__restrict__ scale,
    int tokens, int heads, int logical_blocks,
    int stride_bz_input, int stride_seq_input, int stride_h_input,
    int stride_bz_output, int stride_seq_output, int stride_h_output,
    int stride_bz_mean, int stride_h_mean) {
  int const logical_block = int(blockIdx.x);
  int const head = int(blockIdx.y);
  int const batch = int(blockIdx.z);
  int const row0 = logical_block * Rows;
  int const elements = Rows * HeadDim;

  float local_max = 1.0e-7f;
  for (int linear = int(threadIdx.x); linear < elements;
       linear += int(blockDim.x)) {
    int const row = row0 + linear / HeadDim;
    int const column = linear % HeadDim;
    float value = 0.0f;
    if (row < tokens) {
      value = to_float(input[
          int64_t(batch) * stride_bz_input + int64_t(row) * stride_seq_input
          + int64_t(head) * stride_h_input + column]);
      if constexpr (SubtractMean) {
        value -= to_float(mean[
            int64_t(batch) * stride_bz_mean
            + int64_t(head) * stride_h_mean + column]);
      }
    }
    local_max = fmaxf(local_max, fabsf(value));
  }

  float const amax = block_max(local_max);
  __shared__ float multiplier;
  if (threadIdx.x == 0) {
    scale[(int64_t(batch) * heads + head) * logical_blocks
          + logical_block] = amax / 127.0f;
    multiplier = 127.0f / amax;
  }
  __syncthreads();

  for (int linear = int(threadIdx.x); linear < elements;
       linear += int(blockDim.x)) {
    int const row = row0 + linear / HeadDim;
    int const column = linear % HeadDim;
    if (row < tokens) {
      float value = to_float(input[
          int64_t(batch) * stride_bz_input + int64_t(row) * stride_seq_input
          + int64_t(head) * stride_h_input + column]);
      if constexpr (SubtractMean) {
        value -= to_float(mean[
            int64_t(batch) * stride_bz_mean
            + int64_t(head) * stride_h_mean + column]);
      }
      int const quantized = max(-127, min(127,
          __float2int_rn(value * multiplier)));
      output[int64_t(batch) * stride_bz_output
             + int64_t(row) * stride_seq_output
             + int64_t(head) * stride_h_output + column] =
          static_cast<int8_t>(quantized);
    }
  }
}

template <typename Input, int HeadDim, int Rows, bool SubtractMean>
void launch_quant(
    torch::Tensor const &input, torch::Tensor const &mean,
    torch::Tensor const &output, torch::Tensor const &scale,
    int tokens, int heads, int logical_blocks,
    int stride_seq_input, int stride_h_input,
    int stride_seq_output, int stride_h_output,
    int stride_bz_mean, int stride_h_mean) {
  dim3 const grid(logical_blocks, heads, input.size(0));
  int constexpr threads = 128;
  quantize_int8_kernel<Input, HeadDim, Rows, SubtractMean>
      <<<grid, threads>>>(
          reinterpret_cast<Input const *>(input.data_ptr()),
          SubtractMean ? reinterpret_cast<Input const *>(mean.data_ptr()) : nullptr,
          output.data_ptr<int8_t>(), scale.data_ptr<float>(),
          tokens, heads, logical_blocks,
          int(input.stride(0)), stride_seq_input, stride_h_input,
          int(output.stride(0)), stride_seq_output, stride_h_output,
          stride_bz_mean, stride_h_mean);
  auto const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess,
              "PPU SpargeAttention quantization launch failed: ",
              hggcGetErrorString(error));
}

}  // namespace detail
}  // namespace spargeattention::ppu

namespace {

void validate_quant_tensors(
    torch::Tensor const &input, torch::Tensor const &output,
    torch::Tensor const &scale, torch::Tensor const *mean) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && scale.is_cuda(),
              "PPU quantization tensors must be device tensors");
  TORCH_CHECK(input.get_device() == output.get_device() &&
                  input.get_device() == scale.get_device(),
              "PPU quantization tensors must share one device");
  TORCH_CHECK(input.scalar_type() == torch::kHalf ||
                  input.scalar_type() == torch::kBFloat16,
              "PPU quantization input must be fp16 or bf16");
  TORCH_CHECK(output.scalar_type() == torch::kInt8 &&
                  scale.scalar_type() == torch::kFloat32,
              "PPU quantization output/scale must be int8/fp32");
  TORCH_CHECK(input.dim() == 4 && output.dim() == 4 && scale.dim() == 3,
              "PPU quantization expects rank-4 input/output and rank-3 scale");
  TORCH_CHECK(input.sizes() == output.sizes(),
              "PPU quantization input/output shapes must match");
  TORCH_CHECK(input.stride(3) == 1 && output.is_contiguous() &&
                  scale.is_contiguous(),
              "PPU quantization requires contiguous output/scale and contiguous input head dimension");
  if (mean != nullptr) {
    TORCH_CHECK(mean->is_cuda() && mean->is_contiguous() && mean->dim() == 3,
                "PPU quantization mean must be a contiguous rank-3 device tensor");
    TORCH_CHECK(mean->get_device() == input.get_device(),
                "PPU quantization mean must share the input device");
    TORCH_CHECK(mean->scalar_type() == input.scalar_type(),
                "PPU quantization input and mean dtypes must match");
  }
}

template <int Rows, bool SubtractMean>
void dispatch_quant(
    torch::Tensor const &input, torch::Tensor const &mean,
    torch::Tensor const &output, torch::Tensor const &scale,
    int tensor_layout, int logical_blocks) {
  int const head_dim = int(input.size(3));
  int const tokens = int(input.size(tensor_layout == 0 ? 1 : 2));
  int const heads = int(input.size(tensor_layout == 0 ? 2 : 1));
  int const stride_seq_input = int(input.stride(tensor_layout == 0 ? 1 : 2));
  int const stride_h_input = int(input.stride(tensor_layout == 0 ? 2 : 1));
  int const stride_seq_output = int(output.stride(tensor_layout == 0 ? 1 : 2));
  int const stride_h_output = int(output.stride(tensor_layout == 0 ? 2 : 1));
  int const stride_bz_mean = SubtractMean ? int(mean.stride(0)) : 0;
  int const stride_h_mean = SubtractMean ? int(mean.stride(1)) : 0;

#define SPARGEATTN_PPU_QUANT(INPUT_TYPE, HEAD_DIM)                            \
  spargeattention::ppu::detail::launch_quant<                                \
      INPUT_TYPE, HEAD_DIM, Rows, SubtractMean>(                             \
      input, mean, output, scale, tokens, heads, logical_blocks,             \
      stride_seq_input, stride_h_input, stride_seq_output, stride_h_output,  \
      stride_bz_mean, stride_h_mean)
  if (input.scalar_type() == torch::kHalf) {
    if (head_dim == 64) SPARGEATTN_PPU_QUANT(cutlass::half_t, 64);
    else SPARGEATTN_PPU_QUANT(cutlass::half_t, 128);
  } else {
    if (head_dim == 64) SPARGEATTN_PPU_QUANT(cutlass::bfloat16_t, 64);
    else SPARGEATTN_PPU_QUANT(cutlass::bfloat16_t, 128);
  }
#undef SPARGEATTN_PPU_QUANT
}

}  // namespace

void quant_per_warp_int8_ppu(
    torch::Tensor input, torch::Tensor output, torch::Tensor scale,
    int block_size, int warp_block_size, int tensor_layout) {
  validate_quant_tensors(input, output, scale, nullptr);
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1,
              "tensor_layout must be 0 (NHD) or 1 (HND)");
  TORCH_CHECK(block_size == 128 && warp_block_size == 32,
              "PPU per-warp quantization requires BLKQ=128 and WARPQ=32");
  int const head_dim = int(input.size(3));
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "PPU quantization supports head_dim 64 or 128");
  int const tokens = int(input.size(tensor_layout == 0 ? 1 : 2));
  int const heads = int(input.size(tensor_layout == 0 ? 2 : 1));
  int const blocks = ((tokens + block_size - 1) / block_size)
      * (block_size / warp_block_size);
  TORCH_CHECK(scale.size(0) == input.size(0) && scale.size(1) == heads &&
                  scale.size(2) == blocks,
              "PPU per-warp quantization scale shape mismatch");
  dispatch_quant<32, false>(input, torch::Tensor{}, output, scale,
                            tensor_layout, blocks);
}

void quant_per_block_int8_ppu(
    torch::Tensor input, torch::Tensor mean, torch::Tensor output,
    torch::Tensor scale, int block_size, int tensor_layout) {
  bool const subtract_mean = mean.defined() && mean.numel() != 0;
  validate_quant_tensors(
      input, output, scale, subtract_mean ? &mean : nullptr);
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1,
              "tensor_layout must be 0 (NHD) or 1 (HND)");
  TORCH_CHECK(block_size == 64,
              "PPU K quantization requires BLKK=64");
  int const head_dim = int(input.size(3));
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "PPU quantization supports head_dim 64 or 128");
  int const tokens = int(input.size(tensor_layout == 0 ? 1 : 2));
  int const heads = int(input.size(tensor_layout == 0 ? 2 : 1));
  int const blocks = (tokens + block_size - 1) / block_size;
  TORCH_CHECK(scale.size(0) == input.size(0) && scale.size(1) == heads &&
                  scale.size(2) == blocks,
              "PPU per-block quantization scale shape mismatch");
  if (subtract_mean) {
    TORCH_CHECK(mean.size(0) == input.size(0) && mean.size(1) == heads &&
                    mean.size(2) == head_dim,
                "PPU K mean shape mismatch");
    dispatch_quant<64, true>(input, mean, output, scale,
                             tensor_layout, blocks);
  } else {
    dispatch_quant<64, false>(input, torch::Tensor{}, output, scale,
                              tensor_layout, blocks);
  }
}
