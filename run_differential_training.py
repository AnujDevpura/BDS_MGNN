import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiment_tracking import ExperimentTracker
from pipeline_utils import build_spark


def parse_args():
    p = argparse.ArgumentParser(description="Differential training pipeline for MGNN")
    p.add_argument("--windows-root", default="artifacts/streaming")
    p.add_argument("--global-edge-input", default="artifacts/phase3_final")
    p.add_argument("--output-root", default="artifacts/diff_training")
    p.add_argument("--epochs-per-window", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--min-window-rows", type=int, default=5000)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--enable-wandb", action="store_true")
    return p.parse_args()


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def window_dirs(root: Path):
    return sorted([p for p in root.glob("window_*") if p.is_dir()])


def main():
    args = parse_args()
    windows_root = Path(args.windows_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(
        run_name=f"diff_training_{int(time.time())}",
        output_dir=str(out_root / "tracking"),
        enable_mlflow=args.enable_mlflow,
        enable_wandb=args.enable_wandb,
        project_name="mgnn-bds-diff-train",
    )
    tracker.log_params(vars(args))

    spark = build_spark(
        app_name="MGNN-Diff-Training",
        master=None,
        driver_memory="8g",
        shuffle_partitions=64,
        default_parallelism=32,
    )
    edge_df = spark.read.parquet(args.global_edge_input).select("src", "dst", "weight", "edge_type", "edge_type_id").cache()

    prepared_jobs = []
    wins = window_dirs(windows_root)
    if args.max_windows > 0:
        wins = wins[: args.max_windows]

    for wpath in wins:
        node_df = spark.read.parquet(str(wpath)).select("node_id", "features", "attack_type", "label_multiclass")
        rows = node_df.count()
        if rows < args.min_window_rows:
            continue

        valid_nodes = node_df.select("node_id").distinct().cache()
        edge_sub = (
            edge_df.join(valid_nodes.withColumnRenamed("node_id", "src"), on="src", how="inner")
            .join(valid_nodes.withColumnRenamed("node_id", "dst"), on="dst", how="inner")
        )

        win_root = out_root / wpath.name
        feat_path = win_root / "phase2_window"
        edge_path = win_root / "phase3_window_edges"
        feat_path.mkdir(parents=True, exist_ok=True)
        edge_path.mkdir(parents=True, exist_ok=True)

        node_df.write.mode("overwrite").parquet(str(feat_path))
        edge_sub.write.mode("overwrite").parquet(str(edge_path))
        valid_nodes.unpersist()
        prepared_jobs.append((wpath.name, rows, win_root, feat_path, edge_path))

    spark.stop()

    prev_model = ""
    prev_normalization = ""
    records = []
    for idx, (window_name, rows, win_root, feat_path, edge_path) in enumerate(prepared_jobs):
        pyg_path = win_root / "phase4_pyg"
        model_path = win_root / "phase5_model"
        model_path.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                sys.executable,
                "phase4_export_pyg.py",
                "--feature-input",
                str(feat_path),
                "--edge-input",
                str(edge_path),
                "--output-dir",
                str(pyg_path),
                "--min-nodes",
                "1000",
                "--min-edges",
                "1000",
            ]
        )

        metrics_out = model_path / "run_summary.json"
        cmd = [
            sys.executable,
            "phase5_train_mgnn.py",
            "--input-dir",
            str(pyg_path),
            "--output-dir",
            str(model_path),
            "--epochs",
            str(args.epochs_per_window),
            "--batch-size",
            str(args.batch_size),
            "--metrics-output",
            str(metrics_out),
            "--num-classes",
            "15",
        ]
        if prev_model:
            cmd.extend(["--init-model", prev_model])
            cmd.extend(["--normalization-input", prev_normalization])
        run_cmd(cmd)

        prev_model = str(model_path / "best_model.pt")
        prev_normalization = str(model_path / "feature_normalization.pt")
        summary = json.loads(metrics_out.read_text(encoding="utf-8"))
        summary["window"] = window_name
        summary["window_rows"] = int(rows)
        summary["warm_start"] = bool(idx > 0)
        records.append(summary)
        tracker.log_metrics(
            {
                "window_rows": float(rows),
                "test_macro_f1": float(summary["test_macro_f1"]),
                "runtime_sec": float(summary["runtime_sec"]),
            },
            step=idx,
        )
        tracker.log_artifact(str(metrics_out))

    out_json = out_root / "differential_training_report.json"
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tracker.log_artifact(str(out_json))
    tracker.close()
    print(f"Differential training complete. Report: {out_json}")


if __name__ == "__main__":
    main()
