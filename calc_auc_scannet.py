import numpy as np
import os

def load_gt_poses(pose_dir, frame_names):
    poses = []
    for name in frame_names:
        idx = os.path.splitext(name)[0]
        pose_path = os.path.join(pose_dir, f'{idx}.txt')
        if os.path.exists(pose_path):
            pose = np.loadtxt(pose_path)
            w2c = np.linalg.inv(pose)
            poses.append(w2c[:3, :])
    return poses

def rotation_angle(R):
    cos_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_val))

def translation_angle(t1, t2):
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_val = np.clip(np.dot(t1, t2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_val))

def relative_pose_error(R1, t1, R2, t2):
    R_rel = R1 @ R2.T
    r_err = rotation_angle(R_rel)
    t_err = translation_angle(t1, t2)
    return max(r_err, t_err)

def compute_auc(errors, thresholds=[5, 15, 30]):
    return {f"AUC@{t}": np.mean(np.array(errors) < t) * 100
            for t in thresholds}

def eval_poses(pred_path, gt_poses, name):
    pred = np.load(pred_path)
    n = min(len(pred), len(gt_poses))
    pred = pred[:n]
    gt = gt_poses[:n]
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            R1_p, t1_p = pred[i, :3, :3], pred[i, :3, 3]
            R2_p, t2_p = pred[j, :3, :3], pred[j, :3, 3]
            R_rel_p = R1_p @ R2_p.T
            t_rel_p = t1_p - R_rel_p @ t2_p
            R1_g, t1_g = gt[i][:3, :3], gt[i][:3, 3]
            R2_g, t2_g = gt[j][:3, :3], gt[j][:3, 3]
            R_rel_g = R1_g @ R2_g.T
            t_rel_g = t1_g - R_rel_g @ t2_g
            err = relative_pose_error(R_rel_p, t_rel_p, R_rel_g, t_rel_g)
            errors.append(err)
    auc = compute_auc(errors)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(f"  프레임 수: {n}  페어 수: {len(errors)}")
    for k, v in auc.items():
        print(f"  {k}: {v:.2f}%")
    return auc

if __name__ == "__main__":
    color_48_dir = './litevggt_dataset/scannet/scene0025_01/color_48'
    pose_dir     = './litevggt_dataset/scannet/scene0025_01/pose'
    frame_names  = sorted(os.listdir(color_48_dir))
    print(f"평가 프레임 수: {len(frame_names)}")
    gt_poses = load_gt_poses(pose_dir, frame_names)
    print(f"GT pose 로드 수: {len(gt_poses)}")
    b_auc = eval_poses('./results/baseline_scannet/pred_poses.npy', gt_poses, "Baseline (GA map)")
    p_auc = eval_poses('./results/proposed_scannet/pred_poses.npy', gt_poses, "Proposed (TokenScorer)")
    print(f"\n{'='*40}")
    print("비교 요약")
    print(f"{'='*40}")
    for k in b_auc:
        b, p = b_auc[k], p_auc[k]
        sign = "+" if p - b >= 0 else ""
        print(f"  {k}: baseline={b:.2f}%  proposed={p:.2f}%  ({sign}{p-b:.2f}%)")
