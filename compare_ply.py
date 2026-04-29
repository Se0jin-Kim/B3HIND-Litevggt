import numpy as np
import os
import sys

def load_ply_points(path):
    points = []
    with open(path, 'r') as f:
        header_done = False
        for line in f:
            if line.strip() == 'end_header':
                header_done = True
                continue
            if header_done:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(points)

def chamfer_distance(p1, p2, sample=10000):
    if len(p1) > sample:
        idx = np.random.choice(len(p1), sample, replace=False)
        p1 = p1[idx]
    if len(p2) > sample:
        idx = np.random.choice(len(p2), sample, replace=False)
        p2 = p2[idx]

    diff = p1[:, None, :] - p2[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    d1 = dist.min(axis=1).mean()
    d2 = dist.min(axis=0).mean()
    return d1, d2, (d1 + d2) / 2

def compare(baseline_path, proposed_path, scene_name):
    print(f"\n{'='*50}")
    print(f"Scene: {scene_name}")
    print(f"{'='*50}")

    if not os.path.exists(baseline_path):
        print(f"  [SKIP] baseline 없음: {baseline_path}")
        return
    if not os.path.exists(proposed_path):
        print(f"  [SKIP] proposed 없음: {proposed_path}")
        return

    print("  PLY 로드 중...")
    b = load_ply_points(baseline_path)
    p = load_ply_points(proposed_path)

    print(f"  Baseline  포인트 수: {len(b):,}")
    print(f"  Proposed  포인트 수: {len(p):,}")
    print(f"  포인트 수 차이: {len(p) - len(b):+,}")

    print("  Chamfer Distance 계산 중...")
    d1, d2, cd = chamfer_distance(b, p)
    print(f"  baseline→proposed: {d1:.6f}")
    print(f"  proposed→baseline: {d2:.6f}")
    print(f"  Chamfer Distance:  {cd:.6f}")

    return cd

if __name__ == "__main__":
    scenes = ["scan65", "scan110", "scan114"]
    results = {}

    for scene in scenes:
        b_path = f"./results/baseline_{scene}/recon.ply"
        p_path = f"./results/proposed_{scene}/recon.ply"

        # scan65는 폴더명이 다를 수 있으니 예외 처리
        if scene == "scan65":
            b_path = "./results/baseline/recon.ply"
            p_path = "./results/proposed/recon.ply"

        cd = compare(b_path, p_path, scene)
        if cd is not None:
            results[scene] = cd

    if results:
        print(f"\n{'='*50}")
        print("최종 요약")
        print(f"{'='*50}")
        for scene, cd in results.items():
            print(f"  {scene}: CD = {cd:.6f}")
        print(f"  평균 CD: {np.mean(list(results.values())):.6f}")
