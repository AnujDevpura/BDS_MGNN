import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    def __init__(
        self,
        run_name: str,
        output_dir: str = "artifacts/experiments",
        enable_mlflow: bool = False,
        enable_wandb: bool = False,
        project_name: str = "mgnn-bds",
    ):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow
        self.enable_wandb = enable_wandb
        self.project_name = project_name
        self._mlflow = None
        self._wandb = None
        self._wandb_run = None
        self._init_backends()

    def _init_backends(self):
        if self.enable_mlflow:
            try:
                import mlflow

                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.project_name)
                mlflow.start_run(run_name=self.run_name)
                self._mlflow = mlflow
            except Exception:
                self._mlflow = None
        if self.enable_wandb:
            try:
                import wandb

                self._wandb = wandb
                self._wandb_run = wandb.init(project=self.project_name, name=self.run_name, reinit=True)
            except Exception:
                self._wandb = None
                self._wandb_run = None

    def log_params(self, params: Dict[str, Any]):
        (self.run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        if self._mlflow:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        if self._wandb_run:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        path = self.run_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            rec = {"step": step, **metrics}
            f.write(json.dumps(rec) + "\n")
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        if self._wandb_run:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        if self._mlflow:
            self._mlflow.log_artifact(str(p))
        if self._wandb_run:
            self._wandb.save(str(p))

    def close(self):
        if self._mlflow:
            self._mlflow.end_run()
        if self._wandb_run:
            self._wandb_run.finish()
