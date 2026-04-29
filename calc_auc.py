import numpy as np
from numpy.linalg import qr

def rq_decomp(M):
    """P[:3,:3]에서 K와 R 분리 (RQ decomposition)"""
    Q, R_mat = qr(np.flipud(M).T)
    K = np.flipud(R_mat.T)
    K = np.fliplr(K)
    R = np.flipud(Q.T)
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    return K, R

def load_gt_poses(npz_path, n):
    data = np.load(npz_path)
    poses = []
    for i in range(n):
        P = data[f'world_mat_{i}'][:3, :]   # (3, 4)
        M = P[:, :3]
        K, R = rq_decomp(M)
        t = np.linalg.inv(K) @ P[:, 3]
        poses.append((R, t))
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
    results = {}
    for thr in thresholds:
        results[f"AUC@{thr}"] = np.mean(np.array(errors) < thr) * 100
    return results

def eval_poses(pred_poses_path, gt_npz_path, name):
    pred = np.load(pred_poses_path)   # (S, 3, 4) or (S, 4, 4)
    n = len(pred)
    gt_poses = load_gt_poses(gt_npz_path, n)

    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            R1_p = pred[i, :3, :3]
            t1_p = pred[i, :3, 3]
            R2_p = pred[j, :3, :3]
            t2_p = pred[j, :3, 3]
            R_rel_p = R1_p @ R2_p.T
            t_rel_p = t1_p - R_rel_p @ t2_p

            R1_g, t1_g = gt_poses[i]
            R2_g, t2_g = gt_poses[j]
            R_rel_g = R1_g @ R2_g.T
            t_rel_g = t1_g - R_rel_g @ t2_g

            err = relative_pose_error(R_rel_p, t_rel_p, R_rel_g, t_rel_g)
            errors.append(err)

    auc = compute_auc(errors)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(f"  프레임 수: {n}")
    print(f"  페어 수:   {len(errors)}")
    for k, v in auc.items():
        print(f"  {k}: {v:.2f}%")
    return auc

if __name__ == "__main__":
    gt_npz = "litevggt_dataset/DTU/scan65/cameras.npz"

    baseline_auc = eval_poses(
        "./results/baseline/pred_poses.npy",
        gt_npz, "Baseline (GA map)"
    )
    proposed_auc = eval_poses(
        "./results/proposed/pred_poses.npy",
        gt_npz, "Proposed (TokenScorer)"
    )

    print(f"\n{'='*40}")
    print("비교 요약")
    print(f"{'='*40}")
    for k in baseline_auc:
        b = baseline_auc[k]
        p = proposed_auc[k]
        diff = p - b
        sign = "+" if diff >= 0 else ""
        print(f"  {k}: baseline={b:.2f}%  proposed={p:.2f}%  ({sign}{diff:.2f}%)")
