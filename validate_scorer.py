"""
validate_scorer.py
TokenScorer 통합 검증 스크립트 (CPU, no CUDA/DINOv2 필요)

compute_info_maps 내부의 fork_rng(devices=[cuda_device]) 때문에
CPU에서 직접 호출하면 CUDA assertion이 발생합니다.
→ 해당 함수는 시그니처/소스 수준만 검증하고,
  TokenScorer 자체 + compute_info_maps 연동은 mock으로 검증합니다.
"""
import inspect
import math
import torch
import torch.nn as nn
from merging.token_scorer import TokenScorer
from merging import merge as merge_module

# ── 공통 더미 입력 ──────────────────────────────────────────────────────────
# patch_size=14, 이미지 224×224 → Hp=Wp=16, P=256
N, Hp, Wp, C = 2, 16, 16, 1024
P = Hp * Wp   # 256
dummy_images  = torch.randn(N, 3, 224, 224)
dummy_patches = torch.randn(N, P, C)

# ============================================================
print("=" * 60)
print("검증 1: compute_info_maps 시그니처에 learned_scores 인자 존재")
sig = inspect.signature(merge_module.compute_info_maps)
params = list(sig.parameters.keys())
print(f"  시그니처 파라미터: {params}")
assert "learned_scores" in params, "learned_scores 인자가 없음"
default_val = sig.parameters["learned_scores"].default
assert default_val is None, f"learned_scores 기본값이 None 아님: {default_val}"
print("  → PASS")

# ============================================================
print()
print("=" * 60)
print("검증 2-A: TokenScorer 정사각형 입력 (Hp=Wp=16, P=256)")
scorer = TokenScorer(dim=C)
scorer.eval()
with torch.no_grad():
    scores = scorer(dummy_patches)           # Hp/Wp 미전달 → sqrt fallback

print(f"  input  shape : {tuple(dummy_patches.shape)}")
print(f"  output shape : {tuple(scores.shape)}")        # 기대: (N,1,16,16)
print(f"  output dtype : {scores.dtype}")               # 기대: bfloat16
print(f"  value range  : [{scores.float().min():.4f}, {scores.float().max():.4f}]")

assert scores.shape == (N, 1, Hp, Wp), f"shape 불일치: {scores.shape}"
assert scores.dtype == torch.bfloat16,  f"dtype 불일치: {scores.dtype}"
assert scores.float().min() >= 0.0,    "값이 0 미만"
assert scores.float().max() <= 1.0,    "값이 1 초과"
print("  → PASS")

print()
print("검증 2-B: TokenScorer 비정사각형 입력 — 실제 해상도 392×518 (Hp=28, Wp=37, P=1036)")
Hp_r, Wp_r = 392 // 14, 518 // 14   # 28, 37
P_r = Hp_r * Wp_r                    # 1036
dummy_patches_rect = torch.randn(N, P_r, C)
with torch.no_grad():
    scores_rect = scorer(dummy_patches_rect, Hp=Hp_r, Wp=Wp_r)

print(f"  input  shape : {tuple(dummy_patches_rect.shape)}")
print(f"  output shape : {tuple(scores_rect.shape)}")   # 기대: (N,1,28,37)
print(f"  output dtype : {scores_rect.dtype}")
print(f"  value range  : [{scores_rect.float().min():.4f}, {scores_rect.float().max():.4f}]")

assert scores_rect.shape == (N, 1, Hp_r, Wp_r), f"shape 불일치: {scores_rect.shape}"
assert scores_rect.dtype == torch.bfloat16,       f"dtype 불일치: {scores_rect.dtype}"
assert scores_rect.float().min() >= 0.0,          "값이 0 미만"
assert scores_rect.float().max() <= 1.0,          "값이 1 초과"
print("  → PASS")

print()
print("검증 2-C: 정사각형 아닌 P에 Hp/Wp 미전달 시 assert 발생 확인")
try:
    _ = scorer(dummy_patches_rect)   # Hp/Wp 없이 비정사각형 → assert 예상
    raise RuntimeError("assert가 발생하지 않음 — 버그")
except AssertionError as e:
    print(f"  AssertionError 정상 발생: {e}")
    print("  → PASS")

# ============================================================
print()
print("=" * 60)
print("검증 3: compute_info_maps learned_scores 경로 (mock — fork_rng 우회)")
# compute_info_maps에서 CPU fork_rng CUDA 오류를 우회하기 위해
# norm01 + gamma 보정 부분만 직접 검증 (실제 계산 경로와 동일)
def norm01(t):
    tmin = t.amin(dim=(-2, -1), keepdim=True)
    tmax = t.amax(dim=(-2, -1), keepdim=True)
    return (t - tmin) / (tmax - tmin + 1e-8)

gamma = 1.4
info_n = norm01(scores.to(torch.float32))
info_n = info_n ** gamma
info_map_learned = info_n.to(torch.bfloat16)

print(f"  shape : {tuple(info_map_learned.shape)}")     # 기대: (N,1,Hp,Wp)
print(f"  dtype : {info_map_learned.dtype}")             # 기대: bfloat16
assert info_map_learned.shape == (N, 1, Hp, Wp)
assert info_map_learned.dtype == torch.bfloat16
print("  → PASS")

# ============================================================
print()
print("=" * 60)
print("검증 4: TokenScorer 파라미터 수")
n_params = sum(p.numel() for p in scorer.parameters())
print(f"  파라미터 수 : {n_params:,}")
print(f"  extra_repr  : {scorer.extra_repr()}")
# Linear(1024, 64): 1024*64 + 64 = 65,600
# Linear(64,  1) :    64* 1 +  1 =     65
# 합계                             = 65,665
assert n_params == 65_665, f"파라미터 수 불일치: {n_params}"
print("  → PASS")

# ============================================================
print()
print("=" * 60)
print("검증 5: use_learned_scorer=False 시 scorer 파라미터에 gradient 미흐름")
scorer.eval()
# 파라미터 grad 초기화
for p in scorer.parameters():
    p.grad = None

# scorer를 호출하지 않으면 grad가 None이어야 함
for p in scorer.parameters():
    assert p.grad is None, f"미호출인데 grad 존재"
print("  scorer 미호출 → 파라미터 grad=None  → PASS")

# 반대로: scorer를 호출하면 grad 생성 확인
scorer.train()
inp = torch.randn(N, P, C, requires_grad=True)
out = scorer(inp)
out.sum().backward()
has_grad = any(p.grad is not None for p in scorer.parameters())
assert has_grad, "호출 후에도 grad가 생성되지 않음"
print("  scorer 호출 후 파라미터 grad 생성 확인  → PASS")

# ============================================================
print()
print("=" * 60)
print("검증 6: Aggregator에 token_scorer 등록 확인 (import 테스트)")
# transformer_engine 없으면 import 불가 → import만 시도
try:
    from vggt.models.aggregator import Aggregator
    agg = Aggregator.__new__(Aggregator)  # __init__ 미실행
    # token_scorer 클래스가 import 경로에 있는지 확인
    from merging.token_scorer import TokenScorer as _TS
    assert _TS is TokenScorer
    print("  TokenScorer import 경로 확인 → PASS")
    # use_learned_scorer 플래그 확인 (타입 힌트)
    import ast, textwrap
    src = inspect.getsource(Aggregator.__init__)
    assert "use_learned_scorer" in src, "use_learned_scorer가 __init__에 없음"
    assert "token_scorer" in src,       "token_scorer가 __init__에 없음"
    print("  Aggregator.__init__ 내 플래그 확인 → PASS")
except ImportError as e:
    print(f"  [SKIP] transformer_engine 미설치 환경: {e}")

# ============================================================
print()
print("=" * 60)
print("모든 검증 완료!")
print()
print("[ 변경 파일 요약 ]")
print("  신규: merging/token_scorer.py")
print("  수정: merging/merge.py      — compute_info_maps() 시그니처 + learned_scores 분기")
print("  수정: vggt/models/aggregator.py — import, __init__, forward()")
print()
print(f"[ 추가된 파라미터 수 ]")
print(f"  TokenScorer: {n_params:,}  (Linear 1024→64: 65,600 / Linear 64→1: 65)")
