import math
import torch
import torch.nn as nn


class TokenScorer(nn.Module):
    """
    Learned token importance scorer.

    Replaces the hand-crafted GA map (0.3*var + 0.7*grad) with a lightweight
    MLP that maps DINOv2 patch token features to per-token importance scores.

    Input:  patch_tokens  [N, P, C]  — special tokens excluded
    Output: info_map      [N, 1, Hp, Wp]  — same shape/dtype as compute_info_maps output
    """

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.dim = dim
        hidden = 64
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so the learned scorer starts neutral
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [N, P, C]
        Hp: int | None = None,        # height in patches; if None, inferred via sqrt
        Wp: int | None = None,        # width  in patches; if None, inferred via sqrt
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: [N, P, C]  float32 or bfloat16
            Hp: spatial height in patch units (e.g. H // 14).
                Required for non-square grids (Hp != Wp).
            Wp: spatial width  in patch units (e.g. W // 14).
                Required for non-square grids (Hp != Wp).

        Returns:
            info_map: [N, 1, Hp, Wp]  bfloat16, values in [0, 1]
        """
        N, P, C = patch_tokens.shape

        if Hp is None or Wp is None:
            # Fallback: square grid assumption
            Hp = Wp = int(math.isqrt(P))
            assert Hp * Wp == P, (
                f"TokenScorer: P={P} is not a perfect square. "
                f"Pass explicit Hp and Wp for non-square grids."
            )
        else:
            assert Hp * Wp == P, (
                f"TokenScorer: Hp={Hp} × Wp={Wp} = {Hp*Wp} != P={P}"
            )

        # Score each token: [N, P, C] → [N, P, 1]
        scores = self.scorer(patch_tokens)   # keep scorer in fp32

        # Reshape to spatial map: [N, 1, Hp, Wp]
        scores = scores.permute(0, 2, 1).reshape(N, 1, Hp, Wp)

        # Per-sample min-max normalisation → [0, 1]  (matches norm01 in compute_info_maps)
        s_min = scores.amin(dim=(-2, -1), keepdim=True)
        s_max = scores.amax(dim=(-2, -1), keepdim=True)
        scores = (scores - s_min) / (s_max - s_min + 1e-8)

        return scores.to(torch.bfloat16)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"dim={self.dim}, params={n_params:,}"
