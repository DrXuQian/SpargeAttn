"""Public SpargeAttention APIs with target-specific backend boundaries.

The NVIDIA and PPU extensions are deliberately optional at import time.  A
package built for one target must not load, or silently fall back to, the
other target's private extension.
"""

__all__ = []

try:
    from .core import (
        block_sparse_sage2_attn_cuda,
        spas_sage2_attn_meansim_cuda,
        spas_sage2_attn_meansim_topk_cuda,
        spas_sage_attn_meansim_cuda,
        spas_sage_attn_meansim_topk_cuda,
    )
except (ImportError, OSError) as error:
    CUDA_BACKEND_IMPORT_ERROR = error
else:
    CUDA_BACKEND_IMPORT_ERROR = None
    __all__ += [
        "block_sparse_sage2_attn_cuda",
        "spas_sage2_attn_meansim_cuda",
        "spas_sage2_attn_meansim_topk_cuda",
        "spas_sage_attn_meansim_cuda",
        "spas_sage_attn_meansim_topk_cuda",
    ]

try:
    from .ppu_sparse import (
        RadialAttentionPlan,
        SparseAttentionPlan,
        block_sparse_sage2_attn_ppu,
        make_h3_topk_plan,
        make_radial_plan_from_block_mask,
        make_radial_plan_from_compute_mask,
        make_sol_plan,
        spargeattn_block_sparse_ppu,
        spargeattn_h3_topk_ppu,
        spargeattn_radial_ppu,
        spargeattn_sol_ppu,
    )
except (ImportError, OSError) as error:
    PPU_BACKEND_IMPORT_ERROR = error
else:
    PPU_BACKEND_IMPORT_ERROR = None
    __all__ += [
        "RadialAttentionPlan",
        "SparseAttentionPlan",
        "block_sparse_sage2_attn_ppu",
        "make_h3_topk_plan",
        "make_radial_plan_from_block_mask",
        "make_radial_plan_from_compute_mask",
        "make_sol_plan",
        "spargeattn_block_sparse_ppu",
        "spargeattn_h3_topk_ppu",
        "spargeattn_radial_ppu",
        "spargeattn_sol_ppu",
    ]
