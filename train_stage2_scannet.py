"""
Stage 2 재학습 - ScanNet scene0025_01
L_recon 활성화 버전
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, '.')
from vggt.models.vggt import VGGT
from merging.merge import compute_info_maps
from train_scorer_stage1 import load_model

PATCH_SIZE    = 14
TRAIN_IMG_SIZE = 518
DEPTH_SCALE   = 1000.0   # mm → meter

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# ── Dataset ──────────────────────────────────────────────────────────────
class ScanNetDataset(Dataset):
    def __init__(self, scene_dir, num_frames=8, img_size=518, frame_skip=5):
        self.color_dir = os.path.join(scene_dir, 'color')
        self.depth_dir = os.path.join(scene_dir, 'depth')
        self.pose_dir  = os.path.join(scene_dir, 'pose')
        self.img_size  = img_size
        self.num_frames = num_frames

        all_files = sorted([f for f in os.listdir(self.color_dir)
                            if f.endswith('.jpg')])
        # frame_skip으로 프레임 수 줄이기
        self.files = all_files[::frame_skip]
        print(f"ScanNetDataset: {len(self.files)}프레임 "
              f"(전체 {len(all_files)}에서 {frame_skip}스텝)")

    def __len__(self):
        return max(1, len(self.files) - self.num_frames)

    def _load_image(self, name):
        img = Image.open(os.path.join(self.color_dir, name)).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)   # [3, H, W] float32 [0,1]

    def _load_depth(self, name):
        dep_name = name.replace('.jpg', '.png')
        d = np.array(Image.open(os.path.join(self.depth_dir, dep_name)),
                     dtype=np.float32)
        d = d / DEPTH_SCALE        # mm → meter
        d = Image.fromarray(d).resize((self.img_size, self.img_size),
                                      Image.NEAREST)
        d = torch.from_numpy(np.array(d)).unsqueeze(0)  # [1, H, W]
        return d

    def __getitem__(self, idx):
        names = self.files[idx: idx + self.num_frames]
        images = torch.stack([self._load_image(n) for n in names])  # [S,3,H,W]
        depths = torch.stack([self._load_depth(n) for n in names])  # [S,1,H,W]
        return images, depths

# ── Loss ─────────────────────────────────────────────────────────────────
def scale_shift_invariant_loss(pred, gt):
    """Scale-shift invariant depth L1 loss"""
    mask = (gt > 0.1) & (gt < 10.0) & torch.isfinite(gt)
    if mask.sum() < 100:
        return torch.tensor(0.0, device=pred.device)

    p = pred[mask]
    g = gt[mask]

    # scale-shift 정렬
    scale = (g * p).sum() / (p * p).sum().clamp(min=1e-8)
    p_aligned = scale * p
    return F.l1_loss(p_aligned, g)

# ── LR schedule ──────────────────────────────────────────────────────────
def get_lr_lambda(warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    return fn

def get_tau(step, tau_start, tau_end, total):
    decay = total / 5.0
    return max(tau_end, tau_end + (tau_start - tau_end) * math.exp(-step / decay))

def estimate_keep_ratio(model, Hp, Wp, device):
    pt = getattr(model.aggregator, '_last_patch_tokens', None)
    if pt is None:
        return None
    with torch.no_grad():
        s = model.aggregator.token_scorer(pt, Hp=Hp, Wp=Wp)
        return (s > s.median()).float().mean().item()

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_checkpoint', required=True)
    p.add_argument('--vggt_checkpoint',   required=True)
    p.add_argument('--scene_dir', default='./litevggt_dataset/scannet/scene0025_01')
    p.add_argument('--output_dir', default='./checkpoints/stage2_scannet')
    p.add_argument('--lr',           type=float, default=4e-5)
    p.add_argument('--warmup_steps', type=int,   default=500)
    p.add_argument('--total_steps',  type=int,   default=10000)
    p.add_argument('--batch_size',   type=int,   default=1)
    p.add_argument('--num_frames',   type=int,   default=8)
    p.add_argument('--frame_skip',   type=int,   default=10)
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--lambda_ratio', type=float, default=0.1)
    p.add_argument('--lambda_kl',    type=float, default=0.05)
    p.add_argument('--target_ratio', type=float, default=0.5)
    p.add_argument('--tau_start',    type=float, default=1.0)
    p.add_argument('--tau_end',      type=float, default=0.1)
    p.add_argument('--log_every',    type=int,   default=50)
    p.add_argument('--save_every',   type=int,   default=2000)
    p.add_argument('--device',       default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    model = load_model(args.vggt_checkpoint, device)
    ckpt  = torch.load(args.stage1_checkpoint, map_location=device)
    model.aggregator.token_scorer.load_state_dict(ckpt['scorer_state_dict'])
    print(f"Stage 1 scorer loaded (epoch {ckpt['epoch']})")
    model.aggregator.set_learned_scorer_mode(True)

    # DINOv2 동결, 나머지 학습
    for name, param in model.named_parameters():
        param.requires_grad = 'patch_embed' not in name
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen:,}  Trainable: {n_trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(args.warmup_steps, args.total_steps)
    )

    # Dataset
    dataset = ScanNetDataset(args.scene_dir, args.num_frames,
                             TRAIN_IMG_SIZE, args.frame_skip)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=args.num_workers,
                         drop_last=True)
    data_iter = iter(loader)

    mean = _MEAN.to(device)
    std  = _STD.to(device)
    Hp = Wp = TRAIN_IMG_SIZE // PATCH_SIZE   # 37

    model.train()

    for step in range(args.total_steps):
        tau = get_tau(step, args.tau_start, args.tau_end, args.total_steps)

        try:
            images, depths = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, depths = next(data_iter)

        B, S, C, H, W = images.shape
        images_4d = images.to(device)                          # [B, S, C, H, W]
        depths_4d = depths.to(device)                          # [B, S, 1, H, W]
        images_flat = images_4d.view(B*S, C, H, W)
        depths_flat = depths_4d.view(B*S, 1, H, W)
        mean5 = mean.unsqueeze(0)  # [1,1,3,1,1]
        std5  = std.unsqueeze(0)
        images_normed = ((images_4d.float() - mean5) / std5).to(torch.bfloat16)  # [B,S,C,H,W]
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = \
                model.aggregator(images_normed)
            depth_pred, _ = model.depth_head(
                aggregated_tokens_list, images_normed, patch_start_idx
            )  # [1, S, H, W, 1]

        # depth_pred 형태 맞추기
        depth_pred = depth_pred.squeeze(0).squeeze(-1)  # [S, H, W]
        depth_gt   = depths_flat.squeeze(1)             # [S, H, W]

        # L_recon
        L_recon = scale_shift_invariant_loss(depth_pred, depth_gt)

        # L_ratio
        keep_ratio = estimate_keep_ratio(model, Hp, Wp, device)
        L_ratio = torch.tensor(
            (keep_ratio - args.target_ratio)**2 if keep_ratio else 0.0,
            device=device
        )

        # L_kl_aux
        pt = getattr(model.aggregator, '_last_patch_tokens', None)
        if pt is not None:
            imgs_f = images_flat.float()
            imgs_n = (imgs_f - mean) / std
            with torch.no_grad():
                ga = compute_info_maps(imgs_n, pt.detach())["info_map"]
            ga_flat   = ga.float().reshape(B*S, -1)
            pred_s    = model.aggregator.token_scorer(pt.detach(), Hp=Hp, Wp=Wp)
            pred_flat = pred_s.float().reshape(B*S, -1)
            L_kl = F.kl_div(F.log_softmax(pred_flat, dim=-1),
                             F.softmax(ga_flat.detach(), dim=-1),
                             reduction='batchmean')
        else:
            L_kl = torch.tensor(0.0, device=device)

        L_total = L_recon + args.lambda_ratio * L_ratio + args.lambda_kl * L_kl

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            kr  = f"{keep_ratio:.3f}" if keep_ratio else "N/A"
            lr  = scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | "
                  f"L_total={L_total.item():.5f} | "
                  f"L_recon={L_recon.item():.5f} | "
                  f"L_ratio={L_ratio.item():.5f} | "
                  f"L_kl={L_kl.item():.5f} | "
                  f"keep={kr} | tau={tau:.3f} | lr={lr:.2e}")

        if step % args.save_every == 0 and step > 0:
            path = os.path.join(args.output_dir, f"stage2_step{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': L_total.item(),
                'args': vars(args),
            }, path)
            print(f"저장: {path}")

    path = os.path.join(args.output_dir, "stage2_scannet_final.pt")
    torch.save({'step': args.total_steps,
                'model_state_dict': model.state_dict(),
                'args': vars(args)}, path)
    print(f"최종 저장: {path}")

if __name__ == '__main__':
    main()
