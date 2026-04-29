"""
Stage 2: TokenScorer + Aggregator end-to-end fine-tuning
- DINOv2 backbone(patch_embed)만 동결, 나머지 전체 학습
- Loss: L_total = L_recon + lambda_ratio * L_ratio + lambda_kl * L_kl_aux
    L_recon  : GT depth map이 있으면 scale-and-shift invariant depth L1 loss
               (없으면 0으로 대체)
    L_ratio  : (실제 keep ratio - target ratio)^2
    L_kl_aux : KL(TokenScorer output || GA heuristic map)
               — merge path가 정수 인덱스를 통해 gradient를 차단하므로
                 TokenScorer를 별도 auxiliary KL loss로 학습
- GPU 서버 실행 전제 (CUDA, 권장 A100 이상)

실행 예시 (GT depth 있음):
    python train_scorer_stage2.py \
        --stage1_checkpoint ./checkpoints/stage1/scorer_stage1_epoch5.pt \
        --vggt_checkpoint   /path/to/litevggt.pt \
        --data_dir          /path/to/scannet_scene \
        --output_dir        ./checkpoints/stage2 \
        --total_steps 20000 \
        --lambda_ratio 0.1 \
        --lambda_kl    0.1 \
        --target_ratio 0.5

실행 예시 (GT depth 없음 — L_ratio만 사용):
    python train_scorer_stage2.py \
        --stage1_checkpoint ./checkpoints/stage1/scorer_stage1_epoch5.pt \
        --vggt_checkpoint   /path/to/litevggt.pt \
        --data_dir          /path/to/images_only \
        --output_dir        ./checkpoints/stage2 \
        --lambda_ratio 1.0   # depth GT 없으므로 ratio 가중치 높임

데이터 디렉토리 구조 (GT depth 포함 시):
    data_dir/
        images/   ← jpg / png (8-bit RGB)
            000001.jpg ...
        depths/   ← 16-bit PNG, 픽셀값 / 1000 = 미터  (선택)
            000001.png ...

─────────────────────────────────────────────────────────────────
[eval/criterion.py 호환성에 대한 주석]

eval/criterion.py의 Regr3D_t / ConfLoss_t / MultiLoss는
DUSt3R 방식의 "pts3d_in_other_view" 출력을 기대하며,
VGGT의 실제 출력 키("depth", "world_points")와 직접 호환되지 않는다.

  criterion 필요 형식: gts[i] = {"pts3d": ..., "valid_mask": ..., "camera_pose": ...}
  VGGT 출력 형식:     {"depth": [B,S,H,W,1], "world_points": [B,S,H,W,3], ...}

  따라서 eval/criterion.py를 Stage 2에 직접 적용하려면
  VGGT 출력을 DUSt3R 예측 형식으로 변환하는 어댑터가 필요하다.
  (현재 미구현 — 별도 어댑터 작성 후 L_recon 교체 가능)

  여기서는 GT depth map이 있을 경우 scale-and-shift invariant L1 depth loss를
  L_recon 대용으로 사용한다. GT depth가 없으면 L_recon = 0.

─────────────────────────────────────────────────────────────────
[block.py no_grad를 제거하지 않은 이유]

이전 분석(4단계):
  - token_merge_bipartite2d() 내부에서 info_map은 오직
    topk().indices (torch.long 정수 인덱스) 계산에만 사용된다.
  - 정수 인덱스는 autograd 그래프에 참여하지 않으므로,
    no_grad 블록 안에 있어도 TokenScorer ← info_map 경로의
    gradient가 차단되는 것은 no_grad 때문이 아니라
    정수 인덱싱의 본질적 비미분성 때문이다.
  - block.py의 no_grad를 제거해도 scorer gradient 흐름에 변화 없음.
  → block.py 수정 불필요.

─────────────────────────────────────────────────────────────────
[set_learned_scorer_mode의 allow_scorer_grad 전파를 추가하지 않은 이유]

동일한 이유 (위 block.py 설명 참조).
Stage 2에서 requires_grad는 training script에서 직접 관리한다.
  set_learned_scorer_mode(True) → use_learned_scorer=True 설정 역할만 수행
  requires_grad는 이후 수동 루프로 patch_embed만 동결.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── 레포 루트를 sys.path에 추가 ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from merging.merge import compute_info_maps

# Stage 1에서 정의된 load_model 재사용
from train_scorer_stage1 import load_model

PATCH_SIZE     = 14
TRAIN_IMG_SIZE = 518   # 37 × 14

# ImageNet 정규화 상수 (aggregator.py와 동일)
_RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_RESNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: TokenScorer + Aggregator end-to-end fine-tuning"
    )
    p.add_argument("--stage1_checkpoint", required=True,
                   help="Stage 1에서 저장한 scorer .pt 경로")
    p.add_argument("--vggt_checkpoint",   required=True,
                   help="LiteVGGT pretrained .pt 경로")
    p.add_argument("--data_dir",          required=True,
                   help="학습 데이터 루트 (images/ 필수, depths/ 선택)")
    p.add_argument("--output_dir",        required=True,
                   help="체크포인트 저장 디렉토리")
    p.add_argument("--lr",           type=float, default=4e-5)
    p.add_argument("--warmup_steps", type=int,   default=1_000)
    p.add_argument("--total_steps",  type=int,   default=20_000)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_frames",   type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--lambda_ratio", type=float, default=0.1,
                   help="L_ratio 가중치")
    p.add_argument("--lambda_kl",   type=float, default=0.1,
                   help="L_kl_aux 가중치 (TokenScorer auxiliary KL loss)")
    p.add_argument("--target_ratio", type=float, default=0.5,
                   help="목표 keep ratio (merged 이후 남길 토큰 비율)")
    p.add_argument("--tau_start",    type=float, default=1.0,
                   help="Tau annealing 시작값 (현재 미사용, 향후 Gumbel-softmax용)")
    p.add_argument("--tau_end",      type=float, default=0.1)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--save_every",   type=int,   default=2_000)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Stage 2 중간 체크포인트 경로 (optional)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler (외부 라이브러리 없이)
# ─────────────────────────────────────────────────────────────────────────────
def get_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ─────────────────────────────────────────────────────────────────────────────
# Tau annealing (향후 Gumbel-softmax 방식 도입 시 사용)
# ─────────────────────────────────────────────────────────────────────────────
def get_tau(step: int, tau_start: float, tau_end: float, total_steps: int) -> float:
    decay = total_steps / 5.0
    return max(tau_end, tau_end + (tau_start - tau_end) * math.exp(-step / decay))


# ─────────────────────────────────────────────────────────────────────────────
# Keep ratio 추정
# ─────────────────────────────────────────────────────────────────────────────
def estimate_keep_ratio(
    model: VGGT,
    Hp: int,
    Wp: int,
    device: torch.device,
) -> Optional[float]:
    """
    aggregator.forward()의 부수효과로 저장된 _last_patch_tokens에서
    TokenScorer 출력의 중앙값 초과 비율을 keep ratio로 추정한다.
    forward() 호출 직후에만 유효한 값이 반환된다.
    """
    patch_tokens = getattr(model.aggregator, "_last_patch_tokens", None)
    if patch_tokens is None:
        return None
    with torch.no_grad():
        scores = model.aggregator.token_scorer(
            patch_tokens, Hp=Hp, Wp=Wp
        )                                       # [B*S, 1, Hp, Wp]
        keep = (scores > scores.median()).float().mean()
    return keep.item()


# ─────────────────────────────────────────────────────────────────────────────
# Depth loss (scale-and-shift invariant L1)
# ─────────────────────────────────────────────────────────────────────────────
def scale_shift_invariant_depth_loss(
    pred: torch.Tensor,      # [B, S, H, W] or [B, S, H, W, 1]  float
    gt:   torch.Tensor,      # [B, S, H, W]  float (meters)
    valid: torch.Tensor,     # [B, S, H, W]  bool
) -> torch.Tensor:
    """
    Log-scale invariant depth L1 loss.
    pred: 모델의 depth 예측값 (양수 보장을 위해 내부에서 softplus 적용)
    gt  : GT depth (미터 단위, 0인 픽셀은 invalid)
    valid: True인 픽셀만 loss 계산
    """
    if pred.shape[-1] == 1:
        pred = pred.squeeze(-1)         # [B, S, H, W]

    eps = 1e-6
    pred_log = torch.log(pred.clamp(min=eps))
    gt_log   = torch.log(gt.clamp(min=eps))

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Scale-and-shift alignment per sample
    # median alignment: shift log(pred) so median(pred_log[valid]) == median(gt_log[valid])
    B, S = pred.shape[:2]
    loss_total = torch.tensor(0.0, device=pred.device)
    count = 0

    for b in range(B):
        for s in range(S):
            v = valid[b, s]             # [H, W]
            if v.sum() == 0:
                continue
            pl = pred_log[b, s][v]      # valid 픽셀만
            gl = gt_log[b, s][v]
            # shift: median alignment
            shift = gl.median() - pl.median()
            pl_aligned = pl + shift
            loss_total = loss_total + (pl_aligned - gl).abs().mean()
            count += 1

    return loss_total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (GT depth 선택적 지원)
# ─────────────────────────────────────────────────────────────────────────────
class DepthFrameDataset(Dataset):
    """
    data_dir/images/  : RGB 이미지 (jpg/png)
    data_dir/depths/  : 16-bit depth PNG (선택). 파일명 stem이 images/와 일치해야 함.
                        픽셀값 / 1000 = 미터 (ScanNet / 7Scenes 관례)

    반환값:
        images: [num_frames, 3, H, W]  float32  [0, 1]
        depths: [num_frames, H, W]     float32  (미터, 없으면 zeros)
        has_depth: bool  (depth GT 존재 여부)
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, num_frames: int, img_size: int = TRAIN_IMG_SIZE):
        super().__init__()
        self.num_frames = num_frames
        self.img_size   = img_size

        data_path  = Path(data_dir)
        img_dir    = data_path / "images"
        depth_dir  = data_path / "depths"

        # images/ 디렉토리가 없으면 data_dir 자체를 이미지 루트로 사용
        if not img_dir.exists():
            img_dir = data_path

        all_imgs = sorted(
            p for p in img_dir.rglob("*") if p.suffix.lower() in self._EXTS
        )
        if len(all_imgs) == 0:
            raise FileNotFoundError(f"이미지 파일 없음: {img_dir}")

        # 부족하면 반복 패딩
        if len(all_imgs) < num_frames:
            times = (num_frames // len(all_imgs)) + 1
            all_imgs = (all_imgs * times)[: num_frames + 1]

        self.img_files = all_imgs

        # depth 파일 매핑 (stem 기준)
        self.has_depth = depth_dir.exists()
        if self.has_depth:
            depth_files = {p.stem: p for p in depth_dir.glob("*.png")}
            # 이미지 stem과 depth stem이 1:1 대응되는지 확인
            matched = sum(1 for p in all_imgs if p.stem in depth_files)
            if matched == 0:
                print("[경고] depths/ 파일과 images/ 파일명 stem이 일치하지 않음 → depth 미사용")
                self.has_depth = False
            else:
                self.depth_files = depth_files
                print(f"Depth GT: {matched}/{len(all_imgs)} 파일 매칭")
        else:
            print("[정보] depths/ 디렉토리 없음 → L_recon = 0 (L_ratio만 사용)")

        # 슬라이딩 윈도우
        self.indices = [
            list(range(i, i + num_frames))
            for i in range(len(self.img_files) - num_frames + 1)
        ]

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),           # [0, 1] float32
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(img_size),
        ])

        print(
            f"DepthFrameDataset: {len(self.img_files)} 파일 / "
            f"{len(self.indices)} 시퀀스 / has_depth={self.has_depth}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_depth(self, img_path: Path) -> torch.Tensor:
        """16-bit PNG depth → float32 tensor [H, W] in meters."""
        if not self.has_depth or img_path.stem not in self.depth_files:
            return torch.zeros(self.img_size, self.img_size)
        depth_img = Image.open(self.depth_files[img_path.stem])
        depth_np  = np.array(depth_img, dtype=np.float32) / 1000.0  # mm → m
        depth_t   = torch.from_numpy(depth_np).unsqueeze(0)          # [1, H, W]
        depth_t   = self.depth_transform(depth_t).squeeze(0)         # [H, W]
        return depth_t

    def __getitem__(self, idx: int):
        paths  = [self.img_files[i] for i in self.indices[idx]]
        imgs   = []
        depths = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            imgs.append(self.img_transform(img))
            depths.append(self._load_depth(p))
        return (
            torch.stack(imgs),    # [S, 3, H, W]
            torch.stack(depths),  # [S, H, W]
            self.has_depth,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 모델 준비
# ─────────────────────────────────────────────────────────────────────────────
def setup_model(
    vggt_checkpoint: str,
    stage1_checkpoint: str,
    device: torch.device,
) -> VGGT:
    """
    1. LiteVGGT pretrained 가중치 로드
    2. Stage 1 scorer 가중치 로드
    3. use_learned_scorer = True
    4. DINOv2 backbone(patch_embed)만 동결, 나머지 전체 학습 가능

    set_learned_scorer_mode(True)는 aggregator 내부에서
    use_learned_scorer 플래그를 설정하는 역할만 담당.
    requires_grad는 이후 수동 루프에서 최종 결정한다.
    (allow_scorer_grad 전파 불필요 — block.py 분석 결과 참조)
    """
    model = load_model(vggt_checkpoint, device)

    # Stage 1 scorer 가중치 로드
    s1_ckpt = torch.load(stage1_checkpoint, map_location=device)
    model.aggregator.token_scorer.load_state_dict(s1_ckpt["scorer_state_dict"])
    print(f"Stage 1 scorer loaded  (epoch={s1_ckpt.get('epoch', '?')})")

    # use_learned_scorer 플래그 활성화 (aggregator.py의 info_map 분기 전환)
    model.aggregator.set_learned_scorer_mode(True)

    # --- requires_grad 최종 설정 ---
    # set_learned_scorer_mode(True)가 내부에서 token_scorer 외 전부를 freeze하지만,
    # Stage 2에서는 patch_embed 제외 전체를 학습해야 하므로 아래서 override.
    for name, param in model.named_parameters():
        param.requires_grad = "patch_embed" not in name

    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen    (patch_embed): {n_frozen:,}")
    print(f"Trainable (rest):        {n_trainable:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn (has_depth가 bool이므로 별도 처리)
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    imgs_list, depths_list, has_depth_list = zip(*batch)
    imgs   = torch.stack(imgs_list, dim=0)       # [B, S, 3, H, W]
    depths = torch.stack(depths_list, dim=0)     # [B, S, H, W]
    has_depth = has_depth_list[0]                # bool (batch 내 동일)
    return imgs, depths, has_depth


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ── 모델 ───────────────────────────────────────────────────────────────
    model = setup_model(args.vggt_checkpoint, args.stage1_checkpoint, device)
    model.train()

    # ── Optimizer / Scheduler ──────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(args.warmup_steps, args.total_steps),
    )

    start_step = 0

    # 학습 재개
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"재개: step={start_step}")

    # ── DataLoader ─────────────────────────────────────────────────────────
    dataset = DepthFrameDataset(args.data_dir, args.num_frames, img_size=TRAIN_IMG_SIZE)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    data_iter = iter(loader)

    os.makedirs(args.output_dir, exist_ok=True)

    # Hp, Wp: 고정 해상도 기준 (TRAIN_IMG_SIZE × TRAIN_IMG_SIZE)
    Hp = Wp = TRAIN_IMG_SIZE // PATCH_SIZE   # 37

    # 정규화 상수 device 이동 — 루프 안에서 매 step마다 .to(device) 호출 방지
    resnet_mean = _RESNET_MEAN.to(device)
    resnet_std  = _RESNET_STD.to(device)

    # ── 학습 ───────────────────────────────────────────────────────────────
    for step in range(start_step, args.total_steps):
        tau = get_tau(step, args.tau_start, args.tau_end, args.total_steps)

        # 데이터 로드
        try:
            images, depths, has_depth = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, depths, has_depth = next(data_iter)

        # images: [B, S, 3, H, W]  float32  [0, 1]
        # depths: [B, S, H, W]     float32  meters (0 = invalid)
        images = images.to(device)            # [0,1] float32
        depths = depths.to(device)

        B, S, C, H, W = images.shape

        # ── Forward ────────────────────────────────────────────────────────
        # model.forward()가 aggregator 내부에서 정규화 수행 (aggregator.py:243)
        # → [0,1] 이미지를 그대로 전달
        # use_learned_scorer=True이므로 TokenScorer가 info_map 대신 사용됨
        # aggregator._last_patch_tokens도 이 호출에서 설정됨
        predictions = model(images)
        # predictions 키: "pose_enc", "depth", "depth_conf",
        #                  "world_points", "world_points_conf"

        # ── L_recon (depth loss) ───────────────────────────────────────────
        if has_depth:
            pred_depth = predictions["depth"].float()         # [B, S, H, W, 1]
            gt_depth   = depths.float()                       # [B, S, H, W]
            valid_mask = gt_depth > 1e-3                      # 유효 depth 픽셀
            L_recon = scale_shift_invariant_depth_loss(pred_depth, gt_depth, valid_mask)
        else:
            L_recon = torch.tensor(0.0, device=device)

        # ── L_ratio ────────────────────────────────────────────────────────
        keep_ratio = estimate_keep_ratio(model, Hp, Wp, device)
        if keep_ratio is not None:
            L_ratio = torch.tensor(
                (keep_ratio - args.target_ratio) ** 2,
                device=device, dtype=torch.float32,
            )
        else:
            L_ratio = torch.tensor(0.0, device=device)

        # ── L_kl_aux (TokenScorer auxiliary KL loss) ───────────────────────
        # token_merge_bipartite2d()는 info_map을 정수 인덱스 계산에만 사용하므로
        # L_recon → TokenScorer gradient path가 차단된다.
        # 따라서 TokenScorer를 KL(scorer || GA heuristic)로 별도 학습한다.
        patch_tokens_cached = getattr(model.aggregator, "_last_patch_tokens", None)
        if patch_tokens_cached is not None:
            # GA heuristic map (teacher) — no_grad, float32 변환 후 정규화
            images_flat   = images.view(B * S, C, H, W).float()
            images_normed = (images_flat - resnet_mean) / resnet_std
            with torch.no_grad():
                ga_map = compute_info_maps(
                    images_normed,
                    patch_tokens_cached.detach(),
                )["info_map"]                       # [B*S, 1, Hp, Wp]  bfloat16
            ga_flat = ga_map.float().reshape(B * S, -1)  # [B*S, Hp*Wp]

            # TokenScorer output (student) — gradient 활성화
            pred_scores = model.aggregator.token_scorer(
                patch_tokens_cached.detach(), Hp=Hp, Wp=Wp
            )                                       # [B*S, 1, Hp, Wp]  bfloat16
            pred_flat = pred_scores.float().reshape(B * S, -1)  # [B*S, Hp*Wp]

            L_kl_aux = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(ga_flat.detach(), dim=-1),
                reduction="batchmean",
            )
        else:
            L_kl_aux = torch.tensor(0.0, device=device)

        # ── Total loss ─────────────────────────────────────────────────────
        L_total = L_recon + args.lambda_ratio * L_ratio + args.lambda_kl * L_kl_aux

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── 로깅 ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            lr_now  = scheduler.get_last_lr()[0]
            kr_str  = f"{keep_ratio:.3f}" if keep_ratio is not None else "N/A"
            print(
                f"Step {step:6d} | "
                f"L_total={L_total.item():.5f} | "
                f"L_recon={L_recon.item():.5f} | "
                f"L_ratio={L_ratio.item():.5f} | "
                f"L_kl={L_kl_aux.item():.5f} | "
                f"keep={kr_str} | "
                f"tau={tau:.3f} | "
                f"lr={lr_now:.2e}"
            )

        # ── 저장 ───────────────────────────────────────────────────────────
        if step % args.save_every == 0 and step > 0:
            ckpt_path = os.path.join(args.output_dir, f"stage2_step{step}.pt")
            torch.save(
                {
                    "step":                  step,
                    "model_state_dict":      model.state_dict(),
                    "optimizer_state_dict":  optimizer.state_dict(),
                    "scheduler_state_dict":  scheduler.state_dict(),
                    "loss":                  L_total.item(),
                    "args":                  vars(args),
                },
                ckpt_path,
            )
            print(f"저장: {ckpt_path}")

    # 최종 저장
    final_path = os.path.join(args.output_dir, "stage2_final.pt")
    torch.save(
        {
            "step":                  args.total_steps,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "args":                  vars(args),
        },
        final_path,
    )
    print(f"최종 저장: {final_path}")
    print("Stage 2 완료.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    train(args)
