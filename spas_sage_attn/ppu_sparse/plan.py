"""Canonical block-sparse plans consumed by the PPU SpargeAttention kernel.

The planner owns algorithm policy.  The device kernel only sees an ordered CSR
list of exact KV64 tiles and, optionally, a compact bitset identifying route
blocks that must not also contribute their block-mean summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


KV_BLOCK = 64
SUPPORTED_Q_BLOCKS = (64, 128)
SUPPORTED_ROUTE_BLOCKS = (64, 128)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """Pack a final boolean dimension into signed int32 words."""
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise TypeError("selected-route mask must be rank-2 bool")
    words = _ceil_div(mask.shape[1], 32)
    padded = torch.nn.functional.pad(mask, (0, words * 32 - mask.shape[1]))
    lanes = torch.arange(32, device=mask.device, dtype=torch.int64)
    weights = torch.bitwise_left_shift(torch.ones_like(lanes), lanes)
    packed = (
        padded.reshape(mask.shape[0], words, 32).to(torch.int64)
        * weights
    ).sum(-1)
    return packed.to(torch.int32).contiguous()


def unpack_bits(words: torch.Tensor, count: int) -> torch.Tensor:
    """Inverse of :func:`_pack_bits`, primarily for validation/oracles."""
    if words.dtype != torch.int32 or words.ndim != 2:
        raise TypeError("selected_route_bits must be rank-2 int32")
    shifts = torch.arange(32, device=words.device, dtype=torch.int64)
    unsigned = words.to(torch.int64) & 0xFFFFFFFF
    bits = ((unsigned[..., None] >> shifts) & 1).to(torch.bool)
    return bits.reshape(words.shape[0], -1)[:, :count]


@dataclass(frozen=True)
class SparseAttentionPlan:
    """Device-ready routing contract.

    Rows are flattened in ``(batch, query_head, query_block)`` order.  Exact
    indices are KV64 tile ids.  Summary tensors are HND block means with shape
    ``[batch, kv_heads, route_blocks, head_dim]``.
    """

    exact_row_ptr: torch.Tensor
    exact_kv64: torch.Tensor
    selected_route_bits: torch.Tensor
    key_mean: torch.Tensor
    value_mean: torch.Tensor
    log2_block_counts: torch.Tensor
    batch: int
    query_heads: int
    kv_heads: int
    query_length: int
    kv_length: int
    head_dim: int
    query_block: int
    route_block: int
    algorithm: str
    use_summary: bool
    _admitted: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def query_blocks(self) -> int:
        return _ceil_div(self.query_length, self.query_block)

    @property
    def kv_blocks(self) -> int:
        return _ceil_div(self.kv_length, KV_BLOCK)

    @property
    def route_blocks(self) -> int:
        return _ceil_div(self.kv_length, self.route_block)

    @property
    def rows(self) -> int:
        return self.batch * self.query_heads * self.query_blocks

    @property
    def admitted(self) -> bool:
        return self._admitted

    def validate(self, *, deep: bool = True) -> "SparseAttentionPlan":
        tensors = (
            self.exact_row_ptr,
            self.exact_kv64,
            self.selected_route_bits,
            self.key_mean,
            self.value_mean,
            self.log2_block_counts,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all sparse-plan tensors must share one device")
        if self.query_block not in SUPPORTED_Q_BLOCKS:
            raise ValueError(f"unsupported query block: {self.query_block}")
        if self.route_block not in SUPPORTED_ROUTE_BLOCKS:
            raise ValueError(f"unsupported route block: {self.route_block}")
        if self.route_block % KV_BLOCK:
            raise ValueError("route block must be an integer number of KV64 tiles")
        if self.batch <= 0 or self.query_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("batch and head counts must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.query_length <= 0 or self.kv_length <= 0:
            raise ValueError("sequence lengths must be positive")
        if self.head_dim not in (64, 128):
            raise ValueError("PPU SpargeAttention supports head_dim 64 or 128")
        if self.exact_row_ptr.dtype != torch.int32 or self.exact_row_ptr.ndim != 1:
            raise TypeError("exact_row_ptr must be rank-1 int32")
        if self.exact_kv64.dtype != torch.int32 or self.exact_kv64.ndim != 1:
            raise TypeError("exact_kv64 must be rank-1 int32")
        if self.selected_route_bits.dtype != torch.int32 or self.selected_route_bits.ndim != 2:
            raise TypeError("selected_route_bits must be rank-2 int32")
        if self.exact_row_ptr.numel() != self.rows + 1:
            raise ValueError(
                f"exact_row_ptr has {self.exact_row_ptr.numel()} entries; expected {self.rows + 1}"
            )
        expected_words = _ceil_div(self.route_blocks, 32)
        if tuple(self.selected_route_bits.shape) != (self.rows, expected_words):
            raise ValueError(
                "selected_route_bits shape mismatch: "
                f"got {tuple(self.selected_route_bits.shape)}, expected {(self.rows, expected_words)}"
            )
        expected_means = (self.batch, self.kv_heads, self.route_blocks, self.head_dim)
        if tuple(self.key_mean.shape) != expected_means or tuple(self.value_mean.shape) != expected_means:
            raise ValueError(
                f"summary tensors must have shape {expected_means}; "
                f"got K={tuple(self.key_mean.shape)} V={tuple(self.value_mean.shape)}"
            )
        if self.key_mean.dtype != torch.float16 or self.value_mean.dtype != torch.float16:
            raise TypeError("PPU summary tensors must be fp16")
        if not self.key_mean.is_contiguous() or not self.value_mean.is_contiguous():
            raise ValueError("PPU summary tensors must be contiguous HND block arrays")
        if self.log2_block_counts.dtype != torch.float32 or self.log2_block_counts.ndim != 1:
            raise TypeError("log2_block_counts must be rank-1 fp32")
        if self.log2_block_counts.numel() != self.route_blocks:
            raise ValueError("log2_block_counts length does not match route_blocks")
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("all sparse-plan tensors must be contiguous")

        if not deep:
            return self
        # Deep checks intentionally synchronize when tensors are device-backed.
        # They are admission checks, not part of the timed forward path.
        row_ptr = self.exact_row_ptr.cpu().to(torch.int64)
        indices = self.exact_kv64.cpu().to(torch.int64)
        if row_ptr[0].item() != 0 or row_ptr[-1].item() != indices.numel():
            raise ValueError("CSR endpoints do not cover exact_kv64")
        if bool((row_ptr[1:] < row_ptr[:-1]).any()):
            raise ValueError("exact_row_ptr must be nondecreasing")
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= self.kv_blocks):
            raise ValueError("exact_kv64 contains an out-of-range tile")
        selected = unpack_bits(self.selected_route_bits, self.route_blocks).cpu()
        tiles_per_route = self.route_block // KV_BLOCK
        for row in range(self.rows):
            begin, end = row_ptr[row].item(), row_ptr[row + 1].item()
            values = indices[begin:end]
            if values.numel() != torch.unique(values).numel():
                raise ValueError(f"CSR row {row} contains duplicate KV64 tiles")
            route_ids = torch.div(values, tiles_per_route, rounding_mode="floor")
            if route_ids.numel() and not bool(selected[row, route_ids].all()):
                raise ValueError(f"CSR row {row} names a KV tile whose route bit is clear")
            named_routes = selected[row].nonzero(as_tuple=False).flatten()
            for route in named_routes.tolist():
                first = route * tiles_per_route
                expected = set(range(first, min(first + tiles_per_route, self.kv_blocks)))
                present = set(values.tolist())
                if not expected.issubset(present):
                    raise ValueError(
                        f"selected route {route} in row {row} is missing an exact KV64 tile"
                    )
            if not self.use_summary and begin == end:
                raise ValueError(f"exact-only row {row} cannot be empty")
        if not bool(torch.isfinite(self.log2_block_counts).all()):
            raise ValueError("summary block counts must be finite and positive")
        object.__setattr__(self, "_admitted", True)
        return self


def build_plan(
    *,
    selected_route_mask: torch.Tensor,
    key_mean: torch.Tensor,
    value_mean: torch.Tensor,
    block_counts: torch.Tensor,
    query_length: int,
    kv_length: int,
    query_block: int,
    route_block: int,
    algorithm: str,
    use_summary: bool,
) -> SparseAttentionPlan:
    """Convert a selected route-block mask into the canonical KV64 CSR."""
    if selected_route_mask.dtype != torch.bool or selected_route_mask.ndim != 4:
        raise TypeError("selected_route_mask must be [B,Hq,QBlocks,KRouteBlocks] bool")
    batch, query_heads, query_blocks, route_blocks = selected_route_mask.shape
    if key_mean.ndim != 4:
        raise ValueError("key_mean must be [B,Hkv,KRouteBlocks,D]")
    if value_mean.shape != key_mean.shape:
        raise ValueError("key_mean and value_mean shapes must match")
    if key_mean.shape[0] != batch or key_mean.shape[2] != route_blocks:
        raise ValueError("summary tensors disagree with selected_route_mask")
    if query_blocks != _ceil_div(query_length, query_block):
        raise ValueError("selected mask query-block count is wrong")
    if route_blocks != _ceil_div(kv_length, route_block):
        raise ValueError("selected mask route-block count is wrong")
    if route_block % KV_BLOCK:
        raise ValueError("route block must be divisible by KV64")

    rows = batch * query_heads * query_blocks
    flat_mask = selected_route_mask.reshape(rows, route_blocks)
    tiles_per_route = route_block // KV_BLOCK
    route_ids = torch.arange(route_blocks, device=flat_mask.device, dtype=torch.int64)
    offsets = torch.arange(tiles_per_route, device=flat_mask.device, dtype=torch.int64)
    padded = (route_ids[:, None] * tiles_per_route + offsets[None, :]).reshape(-1)
    valid_tiles = padded < _ceil_div(kv_length, KV_BLOCK)
    expanded = flat_mask[:, :, None].expand(rows, route_blocks, tiles_per_route).reshape(rows, -1)
    expanded = expanded & valid_tiles[None, :]
    lengths = expanded.sum(-1, dtype=torch.int32)
    row_ptr = torch.empty(rows + 1, dtype=torch.int32, device=flat_mask.device)
    row_ptr[0] = 0
    row_ptr[1:] = lengths.cumsum(0)
    exact = padded[None, :].expand(rows, -1)[expanded].to(torch.int32).contiguous()

    counts = block_counts.to(device=flat_mask.device, dtype=torch.float32).contiguous()
    plan = SparseAttentionPlan(
        exact_row_ptr=row_ptr.contiguous(),
        exact_kv64=exact,
        selected_route_bits=_pack_bits(flat_mask),
        key_mean=key_mean.to(torch.float16).contiguous(),
        value_mean=value_mean.to(torch.float16).contiguous(),
        log2_block_counts=counts.log2().contiguous(),
        batch=batch,
        query_heads=query_heads,
        kv_heads=key_mean.shape[1],
        query_length=query_length,
        kv_length=kv_length,
        head_dim=key_mean.shape[-1],
        query_block=query_block,
        route_block=route_block,
        algorithm=algorithm,
        use_summary=bool(use_summary),
    )
    # The builders are the sole production constructors.  Their vectorized
    # expansion is covered exhaustively by the host admission; mark the result
    # admitted without imposing a device-to-host CSR scan on every real plan.
    plan.validate(deep=False)
    object.__setattr__(plan, "_admitted", True)
    return plan


__all__ = [
    "KV_BLOCK",
    "SparseAttentionPlan",
    "build_plan",
    "unpack_bits",
]
