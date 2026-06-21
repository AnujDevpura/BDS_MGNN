import argparse
import subprocess
import sys

from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser(description="Config-driven research runner")
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--run-ablation", action="store_true")
    p.add_argument("--run-streaming", action="store_true")
    p.add_argument("--run-graph-diag", action="store_true")
    p.add_argument("--run-diff-train", action="store_true")
    p.add_argument("--run-realtime-bench", action="store_true")
    p.add_argument("--phase4-input-dir", default="artifacts/phase4_pyg")
    p.add_argument("--model-path", default="artifacts/phase5_model/best_model.pt")
    return p.parse_args()


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if args.run_ablation:
        run_cmd(
            [
                sys.executable,
                "run_ablation_scale.py",
                "--scales",
                ",".join([str(x) for x in cfg.ablation.scales]),
                "--phase2-input",
                cfg.ablation.phase2_input,
                "--phase3-input",
                cfg.ablation.phase3_input,
                "--base-output",
                cfg.ablation.base_output,
                "--epochs",
                str(cfg.ablation.epochs),
                "--batch-size",
                str(cfg.ablation.batch_size),
            ]
            + (["--enable-mlflow"] if cfg.tracking.enable_mlflow else [])
            + (["--enable-wandb"] if cfg.tracking.enable_wandb else [])
        )

    if args.run_streaming:
        run_cmd(
            [
                sys.executable,
                "run_streaming_windows.py",
                "--input",
                cfg.streaming.input,
                "--output-root",
                cfg.streaming.output_root,
                "--window-hours",
                str(cfg.streaming.window_hours),
                "--step-hours",
                str(cfg.streaming.step_hours),
                "--max-windows",
                str(cfg.streaming.max_windows),
            ]
        )

    if args.run_graph_diag:
        run_cmd(
            [
                sys.executable,
                "phase5c_graph_structural_diag.py",
                "--input-dir",
                args.phase4_input_dir,
                "--output",
                "artifacts/metrics/phase5c_graph_structural.json",
            ]
        )

    if args.run_diff_train:
        run_cmd(
            [
                sys.executable,
                "run_differential_training.py",
                "--windows-root",
                cfg.streaming.output_root,
                "--global-edge-input",
                cfg.ablation.phase3_input,
                "--output-root",
                "artifacts/diff_training",
            ]
        )

    if args.run_realtime_bench:
        run_cmd(
            [
                sys.executable,
                "realtime_inference_benchmark.py",
                "--input-dir",
                args.phase4_input_dir,
                "--model-path",
                args.model_path,
                "--output",
                "artifacts/benchmarks/realtime_inference_report.json",
            ]
        )


if __name__ == "__main__":
    main()
