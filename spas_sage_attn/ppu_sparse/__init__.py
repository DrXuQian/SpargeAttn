"""PPU SpargeAttention plans, references, and execution APIs."""

from .plan import KV_BLOCK, SparseAttentionPlan
from .planners import make_h3_topk_plan, make_sol_plan
from .radial import (
    RADIAL_KV_BLOCK,
    RADIAL_QUERY_BLOCK,
    RadialAttentionPlan,
    make_radial_plan_from_block_mask,
    make_radial_plan_from_compute_mask,
    radial_plan_mask,
)
from .reference import (
    quantized_sparse_attention_reference,
    sparse_attention_reference,
)
from .radial_reference import (
    quantized_radial_attention_reference,
    radial_attention_reference,
)
from .api import (
    block_sparse_sage2_attn_ppu,
    spargeattn_block_sparse_ppu,
    spargeattn_h3_topk_ppu,
    spargeattn_radial_ppu,
    spargeattn_sol_ppu,
)

__all__ = [
    "KV_BLOCK",
    "SparseAttentionPlan",
    "RADIAL_KV_BLOCK",
    "RADIAL_QUERY_BLOCK",
    "RadialAttentionPlan",
    "make_h3_topk_plan",
    "make_radial_plan_from_block_mask",
    "make_radial_plan_from_compute_mask",
    "make_sol_plan",
    "radial_plan_mask",
    "sparse_attention_reference",
    "quantized_sparse_attention_reference",
    "quantized_radial_attention_reference",
    "radial_attention_reference",
    "spargeattn_block_sparse_ppu",
    "block_sparse_sage2_attn_ppu",
    "spargeattn_h3_topk_ppu",
    "spargeattn_radial_ppu",
    "spargeattn_sol_ppu",
]
