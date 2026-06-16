import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from plyfile import PlyData
    PLY_AVAILABLE = True
except Exception:
    PLY_AVAILABLE = False


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {ret.returncode}")


def find_method_key(results_dict, iteration):
    preferred = f"ours_{iteration}"
    if preferred in results_dict:
        return preferred
    if len(results_dict) == 1:
        return next(iter(results_dict.keys()))
    raise RuntimeError(
        f"Cannot determine method key for iteration {iteration}. "
        f"Available keys: {list(results_dict.keys())}"
    )


def read_results(model_path, iteration):
    results_path = Path(model_path) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    method_key = find_method_key(data, iteration)
    return method_key, data[method_key]


def get_ply_path(model_path, iteration):
    p = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if p.exists():
        return p
    return None


def get_ply_size_mb(ply_path):
    if ply_path is None or not ply_path.exists():
        return None
    return ply_path.stat().st_size / (1024 * 1024)


def get_vertex_count(ply_path):
    if ply_path is None or not ply_path.exists() or not PLY_AVAILABLE:
        return None
    try:
        ply = PlyData.read(str(ply_path))
        return len(ply["vertex"].data)
    except Exception:
        return None


def fmt(x, digits=4):
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate one trained scene")
    parser.add_argument("-s", "--scene_path", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--render_script", default="render.py", type=str)
    parser.add_argument("--metrics_script", default="metrics.py", type=str)
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--report_name", default=None, type=str)
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    model_path = Path(args.model_path)
    render_script = Path(args.render_script)
    metrics_script = Path(args.metrics_script)

    if not args.skip_render:
        cmd = [
            sys.executable, str(render_script),
            "-s", str(scene_path),
            "-m", str(model_path),
            "--iteration", str(args.iteration),
            "--quiet",
            "--eval",
        ]
        if not args.render_train:
            cmd.append("--skip_train")
        run_cmd(cmd)

    if not args.skip_metrics:
        cmd = [
            sys.executable, str(metrics_script),
            "-m", str(model_path),
        ]
        run_cmd(cmd)

    method_key, metrics = read_results(model_path, args.iteration)

    ply_path = get_ply_path(model_path, args.iteration)
    ply_size_mb = get_ply_size_mb(ply_path)
    vertex_count = get_vertex_count(ply_path)

    print("\n================ EVAL SUMMARY ================")
    print(f"Model path : {model_path}")
    print(f"Scene path : {scene_path}")
    print(f"Iteration  : {args.iteration}")
    print(f"Method key : {method_key}")
    print("")
    print(f"SSIM   : {fmt(metrics.get('SSIM'))}")
    print(f"PSNR   : {fmt(metrics.get('PSNR'))}")
    print(f"LPIPS  : {fmt(metrics.get('LPIPS'))}")
    print(f"PLY MB : {fmt(ply_size_mb, 2)}")
    print(f"Points : {vertex_count if vertex_count is not None else 'N/A'}")
    print("=============================================\n")

    summary = {
        "scene_path": str(scene_path),
        "model_path": str(model_path),
        "iteration": args.iteration,
        "method_key": method_key,
        "metrics": metrics,
        "ply_path": str(ply_path) if ply_path else None,
        "ply_size_mb": ply_size_mb,
        "vertex_count": vertex_count,
    }

    report_name = args.report_name or f"eval_summary_{args.iteration}.json"
    report_path = model_path / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()