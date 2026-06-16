import os
import sys
import subprocess

# ====== 配置区 ======
PROJECT_ROOT = r"E:\semDensGS"  # 项目根目录（convert.py / train.py 所在处）
DATA_ROOT = os.path.join(PROJECT_ROOT, r"data\\3dgs_baseline_dataset\3d-ovs")
DATASETS = ["office_desk"]  # 按顺序执行
CKPT_ITERS = "30000"  # 训练 checkpoint 迭代数


# ===================

def run(cmd_list, cwd=None, log_path=None):
    """封装执行并打印命令；失败时抛异常终止。"""
    # # 确保日志所在的文件夹存在
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # # 使用 w 模式（覆盖）或 a 模式（追加）打开文件
    # with open(log_path, "w", encoding="utf-8") as f:
    #     # stdout=f 将标准输出写入文件
    #     # stderr=subprocess.STDOUT 将标准错误也合并到同一个文件中
    #     subprocess.run(cmd_list, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)
    # print(f">>> 日志已保存至: {log_path}")

    print("\n>>>", " ".join(cmd_list))
    subprocess.run(cmd_list, cwd=cwd, check=True)


def main():
    # 可选：自动发现子目录（如想自动跑所有数据集，把下一行取消注释并注释掉上面的 DATASETS）
    # DATASETS[:] = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]

    for name in DATASETS:
        dataset_dir = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(dataset_dir):
            print(f"[跳过] 未找到数据集目录：{dataset_dir}")
            continue

        # 1) COLMAP 位姿估计
        #run([sys.executable, "convert.py", "-s", dataset_dir], cwd=PROJECT_ROOT)

        # 2) 训练（输出目录与数据集同名）
        out_dir = os.path.join("output", "v6\\3d-ovs\\" + name )
        # log_file = os.path.join(PROJECT_ROOT, out_dir, "train.txt")
        # run([
        #     sys.executable, "train_semDenGS.py",
        #     "-s", dataset_dir,
        #     "-m", out_dir,
        #     #"--checkpoint_iterations", CKPT_ITERS
        # ], cwd=PROJECT_ROOT)

        #3) eval
        run([
            sys.executable, "eval_scene.py",
            "-s", dataset_dir,
            "-m", out_dir,
            "--iteration", "30000"
        ], cwd=PROJECT_ROOT)

        # delete_arr = ["delete_05", "delete_10", "delete_15", "delete_20", "delete_25", "delete_30", "delete_35", "delete_40"]
        # for it in delete_arr:
        #     run([
        #         sys.executable, "eval_scene.py",
        #         "-s", dataset_dir,
        #         "-m", os.path.join(out_dir, it),
        #         "--iteration", "30000"
        #     ], cwd=PROJECT_ROOT)
    run([
        sys.executable, "collect_eval_summaries.py",
        "--root", os.path.join("output", "v6\\3d-ovs\\"),
        "--out_dir", os.path.join("output", "v6\\3d-ovs\\"),
        "--iteration", "30000"
    ], cwd=PROJECT_ROOT)
    print("\n✅ 全部任务完成。")


if __name__ == "__main__":
    main()
