"""
te_fallback.py — Pure-PyTorch drop-in shim for transformer_engine.

When transformer_engine is not installed (e.g. Mac / CPU-only environments),
import this module as `te` and use it transparently:

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling
        HAS_TE = True
    except ImportError:
        from vggt.utils import te_fallback as te
        from vggt.utils.te_fallback import Format, DelayedScaling
        HAS_TE = False

Covered symbols
---------------
te.LayerNorm          → nn.LayerNorm          (same constructor API)
te.Linear             → nn.Linear             (same constructor API)
te.DotProductAttention→ F.scaled_dot_product_attention wrapper (bshd format)
te.LayerNormMLP       → LayerNorm + fc1 + GELU + fc2
te.fp8_autocast       → contextlib.nullcontext (no-op)
Format                → plain namespace with E4M3/E5M2 string attrs
DelayedScaling        → no-op dataclass
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Simple aliases ─────────────────────────────────────────────────────────
LayerNorm = nn.LayerNorm
Linear    = nn.Linear


# ── DotProductAttention ────────────────────────────────────────────────────
class DotProductAttention(nn.Module):
    """
    Mirrors the te.DotProductAttention interface used in attention.py.

    Expected call:
        layer = DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=p,
            attn_mask_type="no_mask",
            qkv_format="bshd",        # batch, seq, heads, dim
        )
        out = layer(query_layer=q, key_layer=k, value_layer=v)
        # q/k/v: [B, N, H, D]  →  out: [B, N, H, D]
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float = 0.0,
        attn_mask_type: str = "no_mask",
        qkv_format: str = "bshd",
    ):
        super().__init__()
        assert qkv_format == "bshd", f"te_fallback only supports qkv_format='bshd', got '{qkv_format}'"
        self.attention_dropout = attention_dropout

    def forward(
        self,
        query_layer: torch.Tensor,   # [B, N, H, D]
        key_layer:   torch.Tensor,   # [B, N, H, D]
        value_layer: torch.Tensor,   # [B, N, H, D]
    ) -> torch.Tensor:               # [B, N, H, D]
        # F.scaled_dot_product_attention expects [B, H, N, D]
        q = query_layer.transpose(1, 2)
        k = key_layer.transpose(1, 2)
        v = value_layer.transpose(1, 2)
        dropout_p = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return out.transpose(1, 2)   # back to [B, N, H, D]


# ── LayerNormMLP ───────────────────────────────────────────────────────────
class LayerNormMLP(nn.Module):
    """
    Mirrors te.LayerNormMLP(hidden_size, ffn_hidden_size).

    te.LayerNormMLP fuses LayerNorm + two-layer MLP (with GELU) into one op.
    This fallback does the same thing in plain PyTorch.

        normMlp = LayerNormMLP(dim, mlp_hidden_dim)
        out = normMlp(x)   # [B, N, dim] → [B, N, dim]
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1  = nn.Linear(hidden_size, ffn_hidden_size)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ── fp8_autocast ───────────────────────────────────────────────────────────
@contextmanager
def fp8_autocast(enabled: bool = False, fp8_recipe=None):
    """No-op context manager replacing te.fp8_autocast."""
    yield


# ── Recipe stubs ───────────────────────────────────────────────────────────
class Format:
    """Stub for transformer_engine.common.recipe.Format."""
    E4M3 = "E4M3"
    E5M2 = "E5M2"


class DelayedScaling:
    """Stub for transformer_engine.common.recipe.DelayedScaling."""
    def __init__(self, fp8_format=None, amax_history_len: int = 16,
                 amax_compute_algo: str = "max"):
        pass
