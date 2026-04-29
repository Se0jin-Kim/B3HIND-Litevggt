import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from merging.merge import soft_merge_with_scores

N, T, C = 2, 256, 1024
r = 128

x = torch.randn(N, T, C)
scores = torch.randn(N, T, requires_grad=True)

merged, unmerge_fn = soft_merge_with_scores(x, scores, r, w=16, h=16)

# 검증 1: shape
assert merged.shape == (N, T - r, C), f"merge shape 오류: {merged.shape}"
print(f"merge shape: {merged.shape} — PASS")

# 검증 2: gradient 흐름 (핵심)
loss = merged.sum()
loss.backward()
assert scores.grad is not None, "scores에 gradient 미도달"
assert scores.grad.abs().sum() > 0
print(f"scores.grad norm: {scores.grad.norm():.4f} — PASS")

# 검증 3: unmerge shape
unmerged = unmerge_fn(merged.detach())
assert unmerged.shape == (N, T, C), f"unmerge shape 오류: {unmerged.shape}"
print(f"unmerge shape: {unmerged.shape} — PASS")

print("모든 검증 통과")
