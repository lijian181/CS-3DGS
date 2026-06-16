import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


DELETE_NAMES = {"delete_05", "delete_10", "delete_15", "delete_20", "delete_25"}


def norm_path(p):
    return str(p).replace("\\", "/")


def infer_info_from_model_path(model_path):
    """
    根据 model_path 自动判断 method / dataset / scene / variant。
    支持类似：
      output/v4/3d-ovs/lawn
      output/v4/3d-ovs/lawn/delete_10
      output/v3_b/3d-ovs/bed
      output/3dgs_baseline_dataset_v2/3d-ovs/bed
      output/3dgs_baseline_dataset_v1/3d-ovs/bed
    """
    mp = norm_path(model_path)
    parts = [x for x in mp.split("/") if x]

    variant = "base"
    if parts and parts[-1] in DELETE_NAMES:
        variant = parts[-1]
        scene = parts[-2] if len(parts) >= 2 else "unknown_scene"
    else:
        scene = parts[-1] if parts else "unknown_scene"

    method = "unknown"
    dataset = "unknown_dataset"

    # 优先根据路径中的关键词判断
    if "v4" in parts:
        method = "v4"
        idx = parts.index("v4")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
    elif "v3_b" in parts:
        method = "v3_b"
        idx = parts.index("v3_b")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
    elif "3dgs_baseline_dataset_v2" in parts:
        method = "v2"
        idx = parts.index("3dgs_baseline_dataset_v2")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
    elif "3dgs_baseline_dataset_v1" in parts:
        method = "v1"
        idx = parts.index("3dgs_baseline_dataset_v1")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
    elif "baseline" in mp.lower() or "gaussian-splatting-origin" in mp.lower():
        method = "baseline"
        for d in ["3d-ovs", "lerf_ovs", "mid_nerf_360_v2"]:
            if d in parts:
                dataset = d
                break
    else:
        for d in ["3d-ovs", "lerf_ovs", "mid_nerf_360_v2"]:
            if d in parts:
                dataset = d
                break

    return method, dataset, scene, variant


def read_summary(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_path = data.get("model_path", "")
    method, dataset, scene, variant = infer_info_from_model_path(model_path)

    metrics = data.get("metrics", {})

    row = {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "variant": variant,
        "iteration": data.get("iteration"),
        "model_path": model_path,
        "scene_path": data.get("scene_path"),
        "SSIM": metrics.get("SSIM"),
        "PSNR": metrics.get("PSNR"),
        "LPIPS": metrics.get("LPIPS"),
        "ply_size_mb": data.get("ply_size_mb"),
        "vertex_count": data.get("vertex_count"),
        "json_path": str(json_path),
    }
    return row


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def add_relative_to_v4_base(rows):
    """
    对每个 dataset/scene，以 v4 base 为参考，计算：
      delta_PSNR
      delta_SSIM
      delta_LPIPS
      point_reduction_vs_v4_base_percent
      size_reduction_vs_v4_base_percent
    """
    base_map = {}

    for r in rows:
        if r["method"] == "v4" and r["variant"] == "base":
            base_map[(r["dataset"], r["scene"])] = r

    for r in rows:
        key = (r["dataset"], r["scene"])
        base = base_map.get(key)

        r["delta_PSNR_vs_v4_base"] = None
        r["delta_SSIM_vs_v4_base"] = None
        r["delta_LPIPS_vs_v4_base"] = None
        r["point_reduction_vs_v4_base_percent"] = None
        r["size_reduction_vs_v4_base_percent"] = None

        if base is None:
            continue

        psnr = safe_float(r["PSNR"])
        ssim = safe_float(r["SSIM"])
        lpips = safe_float(r["LPIPS"])
        b_psnr = safe_float(base["PSNR"])
        b_ssim = safe_float(base["SSIM"])
        b_lpips = safe_float(base["LPIPS"])

        if psnr is not None and b_psnr is not None:
            r["delta_PSNR_vs_v4_base"] = psnr - b_psnr
        if ssim is not None and b_ssim is not None:
            r["delta_SSIM_vs_v4_base"] = ssim - b_ssim
        if lpips is not None and b_lpips is not None:
            r["delta_LPIPS_vs_v4_base"] = lpips - b_lpips

        points = safe_float(r["vertex_count"])
        b_points = safe_float(base["vertex_count"])
        if points is not None and b_points not in [None, 0]:
            r["point_reduction_vs_v4_base_percent"] = (b_points - points) / b_points * 100.0

        size = safe_float(r["ply_size_mb"])
        b_size = safe_float(base["ply_size_mb"])
        if size is not None and b_size not in [None, 0]:
            r["size_reduction_vs_v4_base_percent"] = (b_size - size) / b_size * 100.0

    return rows


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_mean_table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"], r["variant"])].append(r)

    mean_rows = []
    for (dataset, method, variant), items in sorted(groups.items()):
        mean_rows.append({
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "num_scenes": len(items),
            "mean_SSIM": mean([x["SSIM"] for x in items]),
            "mean_PSNR": mean([x["PSNR"] for x in items]),
            "mean_LPIPS": mean([x["LPIPS"] for x in items]),
            "mean_ply_size_mb": mean([x["ply_size_mb"] for x in items]),
            "mean_vertex_count": mean([x["vertex_count"] for x in items]),
            "mean_delta_PSNR_vs_v4_base": mean([x["delta_PSNR_vs_v4_base"] for x in items]),
            "mean_delta_SSIM_vs_v4_base": mean([x["delta_SSIM_vs_v4_base"] for x in items]),
            "mean_delta_LPIPS_vs_v4_base": mean([x["delta_LPIPS_vs_v4_base"] for x in items]),
            "mean_point_reduction_vs_v4_base_percent": mean([x["point_reduction_vs_v4_base_percent"] for x in items]),
            "mean_size_reduction_vs_v4_base_percent": mean([x["size_reduction_vs_v4_base_percent"] for x in items]),
        })

    return mean_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="output",
        help="Root directory to recursively search eval_summary json files."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="Iteration number used in eval_summary filename."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="summary_results",
        help="Directory to save collected tables."
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"eval_summary_{args.iteration}.json"
    json_files = sorted(root.rglob(pattern))

    rows = []
    for jp in json_files:
        try:
            rows.append(read_summary(jp))
        except Exception as e:
            print(f"[WARN] Failed to read {jp}: {e}")

    rows = add_relative_to_v4_base(rows)
    rows = sorted(rows, key=lambda x: (x["dataset"], x["scene"], x["method"], x["variant"]))

    mean_rows = build_mean_table(rows)

    all_csv = out_dir / f"all_eval_summaries_{args.iteration}.csv"
    all_json = out_dir / f"all_eval_summaries_{args.iteration}.json"
    mean_csv = out_dir / f"mean_by_variant_{args.iteration}.csv"
    mean_json = out_dir / f"mean_by_variant_{args.iteration}.json"

    write_csv(all_csv, rows)
    write_json(all_json, rows)
    write_csv(mean_csv, mean_rows)
    write_json(mean_json, mean_rows)

    print(f"\nCollected {len(rows)} eval summaries.")
    print(f"Saved: {all_csv}")
    print(f"Saved: {all_json}")
    print(f"Saved: {mean_csv}")
    print(f"Saved: {mean_json}")

    print("\nMean table:")
    for r in mean_rows:
        print(
            f"{r['dataset']} | {r['method']} | {r['variant']} | "
            f"scenes={r['num_scenes']} | "
            f"PSNR={r['mean_PSNR']:.4f} | "
            f"SSIM={r['mean_SSIM']:.4f} | "
            f"LPIPS={r['mean_LPIPS']:.4f} | "
            f"points_red={r['mean_point_reduction_vs_v4_base_percent']}"
        )


if __name__ == "__main__":
    main()