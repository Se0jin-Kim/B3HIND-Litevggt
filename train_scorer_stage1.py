"""
Stage 1: TokenScorer 사전학습
- VGGT 전체 동결, TokenScorer MLP만 학습
- Teacher signal: 기존 GA map (compute_info_maps heuristic 경로)
- Loss: KL divergence (pred_scores 분포 → ga_map 분포에 정렬)
- GPU 서버 실행 전제 (CUDA)

실행 예시:
    python train_scorer_stage1.py \\
        --checkpoint /path/to/litevggt.pt \\
        --data_dir   /path/to/images \\
        --output_dir ./checkpoints/stage1 \\
        --epochs 5 \\
        --batch_size 4 \\
        --num_frames 8

데이터 디렉토리 구조:
    data_dir/
        img001.jpg
        img002.png
        ...
    이미지 파일들을 num_frames씩 슬라이딩 윈도우로 묶어서 사용.
    파일이 num_frames보다 적으면 반복 샘플링.

전처리 파이프라인 (aggregator.py와 동일):
    - 이미지 [0, 1] 범위로 로드
    - ResNet mean/std 정규화  ← aggregator.py:243 과 동일
    - bfloat16 변환           ← aggregator.py:245 과 동일
    - patch_embed 입력
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from pathlib import Path
from PIL import Image

# ── 레포 루트를 sys.path에 추가 ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from merging.merge import compute_info_maps

# aggregator.py와 동일한 정규화 상수
_RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_RESNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# patch_size 고정 (aggregator.py 기본값)
PATCH_SIZE = 14

# Stage 1 학습 시 이미지 고정 해상도 (patch_size 배수, DataLoader 배치 일관성)
# 518 = 37 × 14 → Hp=37, Wp=37, P=1369
TRAIN_IMG_SIZE = 518


# ─────────────────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: TokenScorer pre-training via KL distillation from GA map"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="LiteVGGT pretrained .pt 경로")
    parser.add_argument("--data_dir",    required=True,
                        help="학습 이미지 디렉토리 (jpg/png 재귀 수집)")
    parser.add_argument("--output_dir",  required=True,
                        help="체크포인트 저장 디렉토리")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_frames",  type=int,   default=8,
                        help="시퀀스당 프레임 수")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--log_every",   type=int,   default=50,
                        help="로그 출력 주기 (step)")
    parser.add_argument("--save_every",  type=int,   default=1,
                        help="체크포인트 저장 주기 (epoch)")
    parser.add_argument("--device",      type=str,   default="cuda")
    # 학습 재개용: scorer 체크포인트 경로 (선택)
    parser.add_argument("--resume_scorer", type=str, default=None,
                        help="Stage 1 중간 체크포인트 경로 (optional)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> VGGT:
    """
    LiteVGGT 체크포인트를 로드한다.
    run_demo.py 로드 방식과 동일 (strict=False).
    token_scorer 파라미터가 없는 구 체크포인트도 허용.
    """
    model = VGGT().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # 체크포인트가 {'model': ...} 형태로 래핑된 경우 대응
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # token_scorer 파라미터 미존재는 정상 (Stage 1에서 새로 학습)
    missing_non_scorer = [k for k in missing if "token_scorer" not in k]
    if missing_non_scorer:
        print(f"[경고] 로드되지 않은 파라미터 ({len(missing_non_scorer)}개):")
        for k in missing_non_scorer[:10]:
            print(f"  {k}")

    model = model.to(torch.bfloat16)
    print(f"모델 로드 완료: {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터 동결
# ─────────────────────────────────────────────────────────────────────────────
def freeze_except_scorer(model: VGGT) -> None:
    """
    token_scorer 파라미터만 학습 가능하게 두고 나머지 전부 동결.
    동결 검증: 예상치 못한 trainable 파라미터가 있으면 즉시 AssertionError.
    """
    for name, param in model.named_parameters():
        param.requires_grad = ("token_scorer" in name)

    unexpected = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "token_scorer" not in n
    ]
    assert len(unexpected) == 0, (
        "예상치 못한 trainable 파라미터:\n" + "\n".join(unexpected)
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}  (TokenScorer only)")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SimpleFrameDataset(Dataset):
    """
    data_dir 하위의 이미지 파일을 num_frames씩 묶어서 반환.

    반환값:
        images: [num_frames, 3, TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]  in [0, 1]
                → 정규화는 학습 루프에서 aggregator.py와 동일하게 수행

    슬라이딩 윈도우 방식으로 샘플 생성.
    총 이미지 수가 num_frames보다 적으면 반복 샘플링(tile)으로 대응.
    """

    # 지원 확장자
    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, num_frames: int):
        super().__init__()
        self.num_frames = num_frames

        # 재귀적으로 이미지 경로 수집
        data_path = Path(data_dir)
        all_files = sorted(
            p for p in data_path.rglob("*")
            if p.suffix.lower() in self._EXTS
        )
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"data_dir={data_dir} 에서 이미지 파일을 찾을 수 없습니다."
            )

        # num_frames보다 적으면 반복해서 채움
        if len(all_files) < num_frames:
            times = (num_frames // len(all_files)) + 1
            all_files = (all_files * times)[:num_frames + 1]

        self.files = all_files

        # 슬라이딩 윈도우로 샘플 인덱스 생성 (stride=1)
        self.indices = [
            list(range(i, i + num_frames))
            for i in range(len(self.files) - num_frames + 1)
        ]

        # 이미지 전처리: [0,1] 범위 유지, 고정 크기 (TRAIN_IMG_SIZE × TRAIN_IMG_SIZE)
        # 정규화는 학습 루프에서 명시적으로 수행 (aggregator.py와 일치시키기 위해)
        self.transform = transforms.Compose([
            transforms.Resize(TRAIN_IMG_SIZE),
            transforms.CenterCrop(TRAIN_IMG_SIZE),
            transforms.ToTensor(),   # → [0, 1] float32
        ])

        print(
            f"Dataset: {len(self.files)} 파일, "
            f"{len(self.indices)} 시퀀스, "
            f"num_frames={num_frames}, "
            f"해상도={TRAIN_IMG_SIZE}×{TRAIN_IMG_SIZE}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_paths = [self.files[i] for i in self.indices[idx]]
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)   # [3, H, W] float32 in [0,1]
            frames.append(img)
        return torch.stack(frames)      # [num_frames, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ── 모델 준비 ──────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # Stage 1: heuristic GA map을 teacher로 사용 (scorer는 호출하지 않음)
    model.aggregator.use_learned_scorer = False

    freeze_except_scorer(model)
    model.train()   # gradient checkpointing 활성화 (aggregator 주석 참고)

    # ── 선택적 학습 재개 ───────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    optimizer = torch.optim.AdamW(
        model.aggregator.token_scorer.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    if args.resume_scorer is not None:
        ckpt = torch.load(args.resume_scorer, map_location="cpu")
        model.aggregator.token_scorer.load_state_dict(ckpt["scorer_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"재개: epoch={start_epoch}, step={global_step}")

    # ── DataLoader ─────────────────────────────────────────────────────────
    dataset = SimpleFrameDataset(args.data_dir, args.num_frames)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,   # 마지막 불완전 배치 제거 (KL 안정성)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 정규화 상수를 device로 이동 (aggregator.py:143 의 register_buffer와 동일 값)
    resnet_mean = _RESNET_MEAN.to(device)
    resnet_std  = _RESNET_STD.to(device)

    # Hp, Wp: 고정 해상도에서 결정 (TRAIN_IMG_SIZE × TRAIN_IMG_SIZE)
    Hp = Wp = TRAIN_IMG_SIZE // PATCH_SIZE   # 518 // 14 = 37

    # ── 학습 ───────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for images in loader:
            # images: [B, num_frames, 3, H, W]  float32  [0, 1]
            B, S, C, H, W = images.shape
            assert H == TRAIN_IMG_SIZE and W == TRAIN_IMG_SIZE, (
                f"예상 해상도 {TRAIN_IMG_SIZE}×{TRAIN_IMG_SIZE}, 실제 {H}×{W}"
            )

            # [B*S, 3, H, W] → device
            images_flat = images.view(B * S, C, H, W).to(device)

            # ── aggregator.py:243-245 와 동일한 전처리 ─────────────────
            # 1) ResNet 정규화 (float32 입력 → float32 출력)
            images_normed = (images_flat - resnet_mean) / resnet_std  # [B*S, 3, H, W]
            # 2) bfloat16 변환 (patch_embed 입력 dtype)
            images_bf16 = images_normed.to(torch.bfloat16)

            # ── Teacher: GA map (no_grad) ──────────────────────────────
            with torch.no_grad():
                # patch_embed: DINOv2 ViT-L (dinov2_vitl14_reg)
                # aggregator.py:250-253 과 동일
                raw = model.aggregator.patch_embed(images_bf16)
                if isinstance(raw, dict):
                    patch_tokens = raw["x_norm_patchtokens"]   # [B*S, P, C]
                else:
                    patch_tokens = raw
                patch_tokens = patch_tokens.to(torch.bfloat16)

                # compute_info_maps: 정규화된 float 이미지 + patch_tokens 입력
                # images_normed는 float32, compute_info_maps 내부에서
                # images_normed.to(float32), patch_tokens.to(float32) 수행 (merge.py:126-127)
                ga_out  = compute_info_maps(images_normed, patch_tokens)
                ga_map  = ga_out["info_map"]          # [B*S, 1, Hp, Wp]  bfloat16
                ga_flat = ga_map.float().flatten(1)   # [B*S, Hp*Wp]  float32

            # ── Student: TokenScorer (gradient 있음) ───────────────────
            # patch_tokens를 detach해서 scorer 파라미터에만 gradient 흐르게 함
            # (patch_embed 파라미터 동결이지만 명시적으로 안전하게 처리)
            pred_map  = model.aggregator.token_scorer(
                patch_tokens.detach(), Hp=Hp, Wp=Wp
            )                                         # [B*S, 1, Hp, Wp]  bfloat16
            pred_flat = pred_map.float().flatten(1)   # [B*S, Hp*Wp]  float32

            # ── KL divergence loss ─────────────────────────────────────
            # KL(P_student || P_teacher)
            #   = Σ P_teacher * (log P_teacher - log P_student)
            # F.kl_div 는 input=log_softmax(student), target=softmax(teacher)
            # reduction='batchmean': loss / B*S
            loss = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(ga_flat.detach(), dim=-1),
                reduction="batchmean",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.aggregator.token_scorer.parameters(), max_norm=1.0
            )
            optimizer.step()
            global_step += 1

            loss_val    = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1

            if global_step % args.log_every == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"KL Loss: {loss_val:.6f}"
                )

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(
            f"── Epoch {epoch+1} 완료 | "
            f"평균 KL Loss: {avg_loss:.6f} | "
            f"총 Step: {global_step}"
        )

        # ── 체크포인트 저장 ────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f"scorer_stage1_epoch{epoch+1}.pt",
            )
            torch.save(
                {
                    "epoch":                epoch + 1,
                    "global_step":          global_step,
                    "scorer_state_dict":
                        model.aggregator.token_scorer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                 avg_loss,
                    "args":                 vars(args),
                },
                ckpt_path,
            )
            print(f"저장: {ckpt_path}")

    print("Stage 1 완료.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    train(args)
