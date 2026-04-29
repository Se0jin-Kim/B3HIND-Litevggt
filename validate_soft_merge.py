import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from merging.merge import soft_merge_with_scores

N, T, C = 2, 256, 1024
r = 128

x = torch.randn(N, T, C)
scores = torch.randn(N, T, requires_grad=True)

merge_fn, unmerge_fn = soft_merge_with_scores(x, scores, r, w=16, h=16)

# 검증 1: merge shape (q만)
q_merged = merge_fn(x)
assert q_merged.shape == (N, T - r, C), f"merge shape 오류: {q_merged.shape}"
print(f"merge shape (q only): {q_merged.shape} — PASS")

# 검증 2: gradient 흐름 (핵심)
loss = q_merged.sum()
loss.backward()
assert scores.grad is not None, "scores에 gradient 미도달"
assert scores.grad.abs().sum() > 0
print(f"scores.grad norm: {scores.grad.norm():.4f} — PASS")

# 검증 3: extra_tensors (q, k, v) — attention.py 호출 방식
scores2 = torch.randn(N, T, requires_grad=True)
merge_fn2, unmerge_fn2 = soft_merge_with_scores(x, scores2, r, w=16, h=16)
k = torch.randn(N, T, C)
v = torch.randn(N, T, C)
q_m, k_m, v_m = merge_fn2(x, mode="mean", extra_tensors=k, extra_tensors_2=v)
assert q_m.shape == (N, T - r, C), f"q_m shape 오류: {q_m.shape}"
assert k_m.shape == (N, T - r, C), f"k_m shape 오류: {k_m.shape}"
assert v_m.shape == (N, T - r, C), f"v_m shape 오류: {v_m.shape}"
print(f"merge shape (q,k,v): {q_m.shape} — PASS")

# 검증 4: unmerge shape
unmerged = unmerge_fn(q_merged.detach())
assert unmerged.shape == (N, T, C), f"unmerge shape 오류: {unmerged.shape}"
print(f"unmerge shape: {unmerged.shape} — PASS")

# 검증 5: 반환 타입 확인 (callable)
assert callable(merge_fn), "merge_fn이 callable이 아님"
assert callable(unmerge_fn), "unmerge_fn이 callable이 아님"
print("merge_fn/unmerge_fn callable — PASS")

print("\n모든 검증 통과")
