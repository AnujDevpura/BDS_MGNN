import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiment_tracking import ExperimentTracker


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation-at-scale experiments for MGNN")
    p.add_argument("--scales", default="100000,300000,550000")
    p.add_argument("--phase2-input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--phase3-input", default="artifacts/phase3_final")
    p.add_argument("--base-output", default="artifacts/ablation")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--enable-wandb", action="store_true")
    return p.parse_args()


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    out_root = Path(args.base_output)
    out_root.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(
        run_name=f"ablation_scale_{int(time.time())}",
        output_dir=str(out_root / "tracking"),
        enable_mlflow=args.enable_mlflow,
        enable_wandb=args.enable_wandb,
        project_name="mgnn-bds-ablation",
    )
    tracker.log_params(
        {
            "scales": scales,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "phase2_input": args.phase2_input,
            "phase3_input": args.phase3_input,
        }
    )

    results = []
    for i, n_nodes in enumerate(scales):
        scale_tag = f"scale_{n_nodes}"
        phase4_out = out_root / scale_tag / "phase4_pyg"
        phase5_out = out_root / scale_tag / "phase5_model"
        phase5_out.mkdir(parents=True, exist_ok=True)
        metrics_out = phase5_out / "run_summary.json"

        run_cmd(
            [
                sys.executable,
                "phase4_export_pyg.py",
                "--feature-input",
                args.phase2_input,
                "--edge-input",
                args.phase3_input,
                "--output-dir",
                str(phase4_out),
                "--max-nodes",
                str(n_nodes),
            ]
        )
        run_cmd(
            [
                sys.executable,
                "phase5_train_mgnn.py",
                "--input-dir",
                str(phase4_out),
                "--output-dir",
                str(phase5_out),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--metrics-output",
                str(metrics_out),
                "--num-classes",
                "15",
            ]
        )

        row = json.loads(metrics_out.read_text(encoding="utf-8"))
        row["scale_nodes_target"] = n_nodes
        results.append(row)
        tracker.log_metrics(
            {
                "nodes": row["nodes"],
                "edges": row["edges"],
                "runtime_sec": row["runtime_sec"],
                "peak_gpu_mem_gb": row["peak_gpu_mem_gb"],
                "test_macro_f1": row["test_macro_f1"],
            },
            step=i,
        )
        tracker.log_artifact(str(metrics_out))

    table_out = out_root / "ablation_results.json"
    table_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tracker.log_artifact(str(table_out))
    tracker.close()
    print(f"Ablation complete. Results: {table_out}")


if __name__ == "__main__":
    main()
