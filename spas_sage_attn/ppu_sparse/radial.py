"""Device-ready RadialAttention masks for the PPU sparse executor.

RadialAttention owns the spatiotemporal masking policy.  This module starts at
the operator boundary used by its SparseSageAttention2 backend: a boolean
block map.  It lowers that map to the fixed Q128/KV64 incremental LUT consumed
by the PPU kernel.  It deliberately does not reimplement Wan frame geometry or
choose dense layers/timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


RADIAL_QUERY_BLOCK = 128
RADIAL_KV_BLOCK = 64
RADIAL_SOURCE_BLOCKS = (64, 128)
RADIAL_SOURCE_GEOMETRIES = ((64, 64), (128, 128), (128, 64))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _expand_identity_mask(
    block_mask: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
) -> torch.Tensor:
    if block_mask.dtype != torch.bool:
        if block_mask.dtype not in (torch.int8, torch.uint8):
            raise TypeError("Radial block_mask must be bool, int8, or uint8")
        if bool(((block_mask != 0) & (block_mask != 1)).any()):
            raise ValueError("integer Radial block_mask values must be binary")
        block_mask = block_mask.bool()
    if block_mask.ndim == 2:
        block_mask = block_mask[None, None]
    elif block_mask.ndim != 4:
        raise ValueError("Radial block_mask must be [Q,K] or [B,H,Q,K]")
    if block_mask.size(0) not in (1, batch):
        raise ValueError(
            f"Radial block_mask batch is {block_mask.size(0)}; expected 1 or {batch}"
        )
    if block_mask.size(1) not in (1, query_heads):
        raise ValueError(
            "Radial block_mask head count is "
            f"{block_mask.size(1)}; expected 1 or {query_heads}"
        )
    return block_mask.expand(batch, query_heads, -1, -1)


def _to_q128_kv64(block_mask: torch.Tensor, source_block: int) -> torch.Tensor:
    """Match RadialAttention's non-SM90 SparseSage mask conversion.

    A source K128 block names both physical KV64 tiles.  A source Q64 pair is
    coarsened by OR because the shipping sparse kernel owns Q128 per CTA.
    """
    if source_block == 128:
        return block_mask.repeat_interleave(2, dim=-1).contiguous()
    if source_block != 64:
        raise ValueError(
            f"Radial source_block must be one of {RADIAL_SOURCE_BLOCKS}, got {source_block}"
        )
    if block_mask.size(-2) & 1:
        block_mask = torch.nn.functional.pad(block_mask, (0, 0, 0, 1))
    shape = (*block_mask.shape[:-2], block_mask.size(-2) // 2, 2, block_mask.size(-1))
    return block_mask.reshape(shape).any(dim=-2).contiguous()


def _mask_to_incremental_lut(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower a Q128/KV64 mask to SparseSageAttention's delta-coded LUT."""
    valid = mask.sum(dim=-1, dtype=torch.int32).contiguous()
    kv_blocks = mask.size(-1)
    absolute = torch.arange(kv_blocks, device=mask.device, dtype=torch.int32)
    absolute = absolute.view(*([1] * (mask.ndim - 1)), kv_blocks).expand_as(mask)
    sentinel = torch.full((), kv_blocks, device=mask.device, dtype=torch.int32)
    ordered = torch.where(mask, absolute, sentinel).sort(dim=-1).values
    lut = torch.zeros_like(ordered, dtype=torch.int32)
    if kv_blocks:
        lut[..., 0] = ordered[..., 0]
        lut[..., 1:] = ordered[..., 1:] - ordered[..., :-1]
        positions = torch.arange(kv_blocks, device=mask.device)
        lut = torch.where(positions < valid[..., None], lut, 0)
    return lut.contiguous(), valid


@dataclass(frozen=True)
class RadialAttentionPlan:
    """Incremental LUT contract consumed by the PPU Radial kernel."""

    block_lut: torch.Tensor
    valid_block_num: torch.Tensor
    batch: int
    query_heads: int
    kv_heads: int
    query_length: int
    kv_length: int
    head_dim: int
    source_query_block: int
    source_kv_block: int
    selected_tiles: int
    _admitted: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def query_blocks(self) -> int:
        return _ceil_div(self.query_length, RADIAL_QUERY_BLOCK)

    @property
    def kv_blocks(self) -> int:
        return _ceil_div(self.kv_length, RADIAL_KV_BLOCK)

    @property
    def admitted(self) -> bool:
        return self._admitted

    def validate(self, *, deep: bool = True) -> "RadialAttentionPlan":
        source_geometry = (self.source_query_block, self.source_kv_block)
        if source_geometry not in RADIAL_SOURCE_GEOMETRIES:
            raise ValueError(f"unsupported Radial source geometry: {source_geometry}")
        if self.batch <= 0 or self.query_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("Radial batch and head counts must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("Radial query heads must be divisible by KV heads")
        if self.query_length <= 0 or self.kv_length <= 0:
            raise ValueError("Radial sequence lengths must be positive")
        if self.head_dim != 128:
            raise ValueError("the first PPU Radial path requires head_dim 128")
        if self.block_lut.dtype != torch.int32 or self.block_lut.ndim != 4:
            raise TypeError("Radial block_lut must be rank-4 int32")
        if self.valid_block_num.dtype != torch.int32 or self.valid_block_num.ndim != 3:
            raise TypeError("Radial valid_block_num must be rank-3 int32")
        expected_lut = (
            self.batch,
            self.query_heads,
            self.query_blocks,
            self.kv_blocks,
        )
        if tuple(self.block_lut.shape) != expected_lut:
            raise ValueError(
                f"Radial block_lut shape is {tuple(self.block_lut.shape)}; expected {expected_lut}"
            )
        if tuple(self.valid_block_num.shape) != expected_lut[:-1]:
            raise ValueError(
                "Radial valid_block_num shape is "
                f"{tuple(self.valid_block_num.shape)}; expected {expected_lut[:-1]}"
            )
        if self.block_lut.device != self.valid_block_num.device:
            raise ValueError("Radial LUT and counts must share one device")
        if not self.block_lut.is_contiguous() or not self.valid_block_num.is_contiguous():
            raise ValueError("Radial LUT and counts must be contiguous")
        if not deep:
            return self

        counts = self.valid_block_num.to(torch.int64)
        if bool((counts <= 0).any()) or bool((counts > self.kv_blocks).any()):
            raise ValueError("every Radial row must name 1..kv_blocks exact tiles")
        if int(counts.sum().item()) != self.selected_tiles:
            raise ValueError("Radial valid-count coverage differs from its mask authority")
        absolute = self.block_lut.to(torch.int64).cumsum(dim=-1)
        positions = torch.arange(self.kv_blocks, device=absolute.device)
        live = positions < counts[..., None]
        if bool((absolute[live] < 0).any()) or bool((absolute[live] >= self.kv_blocks).any()):
            raise ValueError("Radial LUT resolves an out-of-range KV64 tile")
        if self.kv_blocks > 1:
            adjacent_live = live[..., 1:]
            if bool(((absolute[..., 1:] <= absolute[..., :-1]) & adjacent_live).any()):
                raise ValueError("Radial LUT must resolve strictly increasing KV64 tiles")
        object.__setattr__(self, "_admitted", True)
        return self


def _make_plan(
    compute_mask: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    kv_length: int,
    head_dim: int,
    source_query_block: int,
    source_kv_block: int,
) -> RadialAttentionPlan:
    lut, valid = _mask_to_incremental_lut(compute_mask)
    plan = RadialAttentionPlan(
        block_lut=lut,
        valid_block_num=valid,
        batch=int(batch),
        query_heads=int(query_heads),
        kv_heads=int(kv_heads),
        query_length=int(query_length),
        kv_length=int(kv_length),
        head_dim=int(head_dim),
        source_query_block=int(source_query_block),
        source_kv_block=int(source_kv_block),
        selected_tiles=int(valid.to(torch.int64).sum().item()),
    )
    plan.validate(deep=True)
    return plan


def make_radial_plan_from_compute_mask(
    block_mask: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    kv_length: int,
    head_dim: int = 128,
) -> RadialAttentionPlan:
    """Consume Radial's existing non-SM90 Q128/KV64 ``mask_id`` directly."""
    mask = _expand_identity_mask(
        block_mask, batch=batch, query_heads=query_heads
    )
    expected = (
        _ceil_div(query_length, RADIAL_QUERY_BLOCK),
        _ceil_div(kv_length, RADIAL_KV_BLOCK),
    )
    if tuple(mask.shape[-2:]) != expected:
        raise ValueError(
            f"Radial block_mask tail is {tuple(mask.shape[-2:])}; "
            f"expected compute geometry {expected}"
        )
    return _make_plan(
        mask.contiguous(),
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_length=query_length,
        kv_length=kv_length,
        head_dim=head_dim,
        source_query_block=RADIAL_QUERY_BLOCK,
        source_kv_block=RADIAL_KV_BLOCK,
    )


def make_radial_plan_from_block_mask(
    block_mask: torch.Tensor,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    kv_length: int,
    head_dim: int = 128,
    source_block: int = 128,
) -> RadialAttentionPlan:
    """Lower a square Radial policy mask without reproducing that policy."""
    if source_block not in RADIAL_SOURCE_BLOCKS:
        raise ValueError(
            f"Radial source_block must be one of {RADIAL_SOURCE_BLOCKS}, got {source_block}"
        )
    source_q = _ceil_div(query_length, source_block)
    source_k = _ceil_div(kv_length, source_block)
    mask = _expand_identity_mask(
        block_mask, batch=batch, query_heads=query_heads
    )
    if tuple(mask.shape[-2:]) != (source_q, source_k):
        raise ValueError(
            f"Radial block_mask tail is {tuple(mask.shape[-2:])}; "
            f"expected {(source_q, source_k)} for source_block={source_block}"
        )
    compute_mask = _to_q128_kv64(mask, source_block)
    expected_compute = (
        _ceil_div(query_length, RADIAL_QUERY_BLOCK),
        _ceil_div(kv_length, RADIAL_KV_BLOCK),
    )
    compute_mask = compute_mask[..., : expected_compute[0], : expected_compute[1]]
    if tuple(compute_mask.shape[-2:]) != expected_compute:
        raise AssertionError(
            "Radial source-to-compute block conversion produced the wrong extent"
        )
    return _make_plan(
        compute_mask,
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_length=query_length,
        kv_length=kv_length,
        head_dim=head_dim,
        source_query_block=source_block,
        source_kv_block=source_block,
    )


def radial_plan_mask(plan: RadialAttentionPlan) -> torch.Tensor:
    """Recover the admitted Q128/KV64 mask for tests and diagnostics."""
    plan.validate(deep=True)
    absolute = plan.block_lut.to(torch.int64).cumsum(dim=-1)
    positions = torch.arange(plan.kv_blocks, device=absolute.device)
    live = positions < plan.valid_block_num[..., None]
    # scatter_add makes unused zero-valued LUT slots harmless even when their
    # clamped indices alias a live final block.  Plain scatter would allow a
    # later False source to erase that live entry.
    hits = torch.zeros(
        (*plan.valid_block_num.shape, plan.kv_blocks),
        dtype=torch.int32,
        device=plan.block_lut.device,
    )
    hits.scatter_add_(
        -1,
        absolute.clamp(0, plan.kv_blocks - 1),
        live.to(torch.int32),
    )
    return hits != 0


__all__ = [
    "RADIAL_KV_BLOCK",
    "RADIAL_QUERY_BLOCK",
    "RadialAttentionPlan",
    "make_radial_plan_from_block_mask",
    "make_radial_plan_from_compute_mask",
    "radial_plan_mask",
]
