import argparse
import json
from datetime import timedelta
from pathlib import Path

from pyspark.sql import functions as F

from pipeline_utils import build_spark, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="Create time-evolving graph snapshots")
    p.add_argument("--input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--output-root", default="artifacts/streaming")
    p.add_argument("--window-hours", type=int, default=6)
    p.add_argument("--step-hours", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="8g")
    p.add_argument("--shuffle-partitions", type=int, default=64)
    p.add_argument("--default-parallelism", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spark = build_spark(
        app_name="MGNN-Streaming-Windows",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    df = spark.read.parquet(args.input).select("node_id", "event_time", "attack_type", "label_multiclass", "label_binary", "features")
    df = df.where(F.col("event_time").isNotNull()).cache()

    bounds = df.select(
        F.min("event_time").alias("min_t"),
        F.max("event_time").alias("max_t"),
    ).first()
    min_t = bounds["min_t"]
    max_t = bounds["max_t"]
    if min_t is None or max_t is None:
        raise ValueError("event_time bounds not found")

    snapshot_stats = []
    for i in range(args.max_windows):
        start_t = min_t + timedelta(hours=i * args.step_hours)
        end_t = start_t + timedelta(hours=args.window_hours)
        if start_t > max_t:
            break
        win_df = df.where((F.col("event_time") >= F.lit(start_t)) & (F.col("event_time") < F.lit(end_t)))
        cnt = win_df.count()
        if cnt == 0:
            continue
        out_path = out_root / f"window_{i:03d}"
        win_df.write.mode("overwrite").parquet(str(out_path))
        snapshot_stats.append({"window_id": i, "rows": int(cnt), "path": str(out_path)})

    metrics_path = out_root / "streaming_windows_metrics.json"
    metrics_path.write_text(json.dumps(snapshot_stats, indent=2), encoding="utf-8")
    write_metrics_json(
        str(out_root / "streaming_run_meta.json"),
        {
            "phase": "streaming_windows",
            "input": args.input,
            "output_root": str(out_root),
            "windows_created": len(snapshot_stats),
            "max_windows_requested": args.max_windows,
            "window_hours": args.window_hours,
            "step_hours": args.step_hours,
        },
    )
    spark.stop()
    print(f"Created {len(snapshot_stats)} windows. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
