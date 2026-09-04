# PPU SpargeAttention

## Dedicated RadialAttention path

RadialAttention is not executed by translating its mask into the generic
H3/Sol CSR+summary kernel.  Its production backend is SparseSageAttention2,
so the PPU implementation has a separate operator with the same execution
boundary:

- one delta-coded `block_lut[B,H,Q128,KV64]` and one
  `valid_block_num[B,H,Q128]` tensor;
- `Q128/KV64`, D128, eight warps for the FP16-V specialization (one Q16 tile
  per warp);
- Q loaded once for the LUT row, K/V advanced by LUT deltas and pipelined,
  with online softmax and PV in the same loop;
- no summary branch and no Radial/Wan routing policy inside the kernel.

`make_radial_plan_from_compute_mask` directly accepts the non-SM90
SparseSageAttention2 `mask_id` that Radial already produces.  For integrations
that still hold Radial's square policy mask,
`make_radial_plan_from_block_mask` reproduces the published conversion:
K128 columns expand to two KV64 columns, while adjacent Q64 rows are OR-paired
into Q128 rows.  Both conversions and the LUT roundtrip are independently
host-anchored.

For the current Radial call site, `block_sparse_sage2_attn_ppu` is the direct
adapter: it accepts the same Q/K/V plus converted `mask_id`, constructs the
admitted plan, and invokes the dedicated kernel.  It rejects nonzero dropout
instead of silently changing semantics.

```python
from spas_sage_attn import (
    make_radial_plan_from_compute_mask,
    spargeattn_radial_ppu,
)

plan = make_radial_plan_from_compute_mask(
    converted_mask,  # [B, Hq, ceil(Nq/128), ceil(Nkv/64)]
    batch=q.shape[0], query_heads=q.shape[1], kv_heads=k.shape[1],
    query_length=q.shape[2], kv_length=k.shape[2], head_dim=128,
)
out = spargeattn_radial_ppu(q, k, v, plan, tensor_layout="HND")
```

The PPU specialization deliberately differs from current NVIDIA
SparseSageAttention2 in two hardware-facing details: V remains FP16 instead
of FP8, and PPU AIU cube loads replace CUDA cooperative-copy/PTX machinery.
The mask semantics, work decomposition and online recurrence are aligned;
performance parity is a device measurement, not inferred from that alignment.
The first shipping boundary is forward-only BF16 Q/K/V, BF16 output, D128,
non-causal, and no LSE/PV-threshold result.

The performance runner compares prequantized Radial core time with the
prequantized dense core on the same operands.  It reports
`density_efficiency = speedup * selected_KV64_density`; the preregistered
classification is `ALIGNED >= 0.85`, `PARTIAL >= 0.70`, otherwise
`NOT-ALIGNED`.  Its built-in mask is an explicitly labelled irregular
operator fixture.  A real Radial `mask_id` can replace it without code changes:

```bash
RADIAL_MASK=/workspace/wan-radial-mask.pt RADIAL_MASK_KIND=compute \
  OUT=/workspace/spargeattn-radial-real \
  bash tools/run_ppu_sparse_box.sh
```

Local admission builds the exact PPU TU and reads its generated code:

```bash
PPU_SDK=/path/to/PPU_SDK \
  bash dev/ppu_sparse/build_ppu.sh
```

The current generated specialization is stack-free, uses 188 vector
registers, and contains the exact 16 INT8 QK MMAs, 32 FP32-accumulating PV
MMAs and four accumulator-to-PV bridge MMAs implied by one Q128/KV64 visit.
Those are artifact checks, not source assertions.

The first PPU sparse forward has one executor and two routing front ends.  It
does not turn either community algorithm into a dense boolean mask.

## Canonical plan

`SparseAttentionPlan` owns an ordered CSR list of exact KV64 tiles plus a
bitset of selected route blocks.  The device executor performs exact INT8-QK / 
FP16-PV attention over the CSR list.  When summary correction is enabled it
then visits the contiguous block means, masks means already represented by an
exact tile, adds `log2(valid_tokens)`, and continues the same online softmax.

The kernel does not know whether a plan came from H3 top-k or Sol:

| planner | Q route block | K route block | exact lowering |
|---|---:|---:|---|
| H3 top-k | 128 | 128 | one selected K128 becomes two KV64 tiles |
| Sol | 64 | 64 | one selected K64 becomes one KV64 tile |

The H3 planner reproduces block-mean top-k routing.  The Sol planner reproduces
the diagonal/exact threshold choice, forced adjacent blocks, and explicit KV
and query sink ranges.  Both canonicalize exact execution into increasing KV
order so repeated launches have a fixed floating-point order.

## Deliberate first-shipping boundary

The native sparse ABI is forward-only and fail-closes unless all of these are
true:

- BF16 Q/K/V and BF16 output;
- head dimension 128;
- non-causal execution described by an admitted plan;
- no LSE result;
- Q blocks of 64 (Sol) or 128 (H3), and KV compute blocks of 64.

The generic host planner can model other head dimensions, but successful host
algebra is not device support.  The SDK census therefore contains exactly four
sparse specializations, not a speculative cross product.

PPU code generation currently assigns 256 vector registers and an 8-byte
private frame to each summary specialization.  Exact-only specializations and
all pre-existing dense/quantization kernels remain stack-free.  This is an
explicit ACU postcondition: a frame in metadata is not yet evidence of actual
private-memory traffic, but it is not hidden as `spill=0` either.

## Admission

Host admission proves plan algebra independently of the device executor.  It
also runs three expected-red identity mutations and rejects any box runner
that contains a compiler, linker, or disassembler entry point:

```bash
bash dev/ppu_sparse/run_local_gates.sh
```

The checked-in PPU extension is compiled locally from the real `-arch=ppu_10`
source graph.  Its manifest binds the binary hash to the source hashes,
actlize revision, Python/Torch ABI, SDK release, and the locally measured
resource census.  Separate Torch 2.8 and 2.9 artifacts are selected
automatically from measured Python/Torch/CXX11 ABI identity.  The box runner
is execution-only: it verifies that identity, copies the matching prebuilt
extension into the package, and never invokes `hgcc`, `build_ext`, or a
disassembler.  It then requires:

1. full-density Q128 CSR is raw-bit identical to dense PPU SageAttention;
2. full-density Q64 CSR is raw-bit identical to the same dense path;
3. H3 top-k summary agrees with an oracle made from the exact quantized
   operands consumed by the kernel;
4. Sol diagonal-summary agrees with that same independent oracle;
5. tail, GQA, HND/NHD and repeated-launch fingerprints pass;
6. a mutated, unadmitted plan is rejected before launch.

```bash
OUT=/workspace/spargeattn-ppu-sparse-run \
  bash tools/run_ppu_sparse_box.sh
```

`PPU_RUNTIME_DIR` may override the default `/usr/local/PPU_SDK/lib`; it names
runtime libraries only and is not a compiler path.  A source, ABI, submodule,
or binary mismatch fails before any device launch.

## First PPU device verdict

On 2026-09-04, commit `8ea96fc` and the Torch 2.9 artifact
`7711a66e1c3980a6d16be3df58abde6cdaea19b4c06910d33ff3f9f3adc0af41`
passed the complete box admission at `B=1, H=Hkv=16, N=4096, D=128`.
Both full-density sparse geometries were raw-bit identical to dense.  H3 and
Sol agreed with the independent quantized oracle to maximum absolute errors
of `1.2536e-4` and `1.2467e-4`, respectively, and repeated-launch fingerprints
were stable.

| measured region | median (us) | speedup vs dense |
|---|---:|---:|
| dense quantization + core | 584.048 | 1.000x |
| H3 prequantized core, 25% exact density | 152.956 | 3.818x |
| H3 preplanned quantization + core | 347.368 | 1.681x |
| Sol prequantized core, 10.9375% exact density | 100.520 | 5.810x |
| Sol preplanned quantization + core | 292.352 | 1.998x |

The semantic-reference planners cost 871.080 us (H3) and 982.000 us (Sol),
so planning on every invocation reverses both wins to 0.479x and 0.458x.  The
hard per-invocation planning budgets for merely beating dense are therefore
236.680 us for H3 and 291.696 us for Sol.  These numbers define the budget for
an integrating framework, not work owned by this repository: production route
generation may come from ComfyUI/Triton or another policy layer.  This project
owns the actlize/CUTLASS executor beginning at the admitted
`SparseAttentionPlan` boundary.  The Python planners remain semantic oracles
and admission fixtures; they are not claimed as a production routing path.
The exact-tile density is not a complete work fraction because summary rows
also execute.

Performance output keeps three costs separate: planner, prequantized sparse
core, and preplanned quantization-plus-core.  The current Python/Torch Sol
planner materializes its route score matrix; it is the semantic reference and
first device baseline, not the final long-sequence routing implementation.

ComfyUI's Sol-Attn production path is a Triton implementation that performs
routing inside its forward kernel.  It is intentionally not vendored or
reimplemented here.  Framework integration is expected to lower its routing
decision into this executor's explicit metadata contract, or select a separate
Triton backend outside the CUTLASS operator package.

## Sources of semantics

- H3 top-k and summary-row semantics:
  `Occipital-Labs/h3-sparse-attn` (MIT).
- Sol threshold, neighbour, sink and summary semantics:
  `Saganaki22/ComfyUI-sol-attn`.

This implementation reuses neither project's NVIDIA/Triton instruction code.
Only their forward algorithms are transcribed onto the existing actlize PPU
AIU primitives.
