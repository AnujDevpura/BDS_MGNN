import argparse

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pipeline_utils import build_spark, class_distribution, validate_min_rows, write_metrics_json


TARGET_COUNTS = {
    "benign": 150000,
    "dos_hulk": 100000,
    "ddos": 80000,
    "portscan": 80000,
    "dos_goldeneye": 30000,
    "ftp_patator": 25000,
    "ssh_patator": 25000,
    "dos_slowloris": 20000,
    "dos_slowhttptest": 20000,
    "bot": 9000,
    "web_bruteforce": 7000,
    "web_xss": 3000,
    "infiltration": 1000000,
    "web_sql_injection": 1000000,
    "heartbleed": 1000000,
}


def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 2B - stratified canonical node universe")
    p.add_argument("--input", default="artifacts/phase2_features")
    p.add_argument("--output", default="artifacts/phase2_sampled_500k")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase2b_metrics.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="8g")
    p.add_argument("--shuffle-partitions", type=int, default=64)
    p.add_argument("--default-parallelism", type=int, default=32)
    p.add_argument("--output-format", choices=["parquet"], default="parquet")
    p.add_argument("--output-mode", default="overwrite")
    p.add_argument("--min-output-rows", type=int, default=100000)
    return p.parse_args()


def main():
    args = parse_args()

    spark = build_spark(
        app_name="MGNN-Phase2B-StratifiedSampling",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    print("Loading Phase 2 feature dataset...")
    df = spark.read.parquet(args.input)

    required_cols = ["node_id", "event_time", "attack_type", "label_multiclass", "label_binary", "features"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.where(F.col("attack_type").isin(list(TARGET_COUNTS))).dropDuplicates(["node_id"]).cache()
    class_counts_df = df.groupBy("attack_type").count().cache()
    class_counts = {r["attack_type"]: int(r["count"]) for r in class_counts_df.collect()}

    fractions = {}
    for cls, cnt in class_counts.items():
        target = TARGET_COUNTS.get(cls, cnt)
        fractions[cls] = min(1.0, float(target) / float(cnt)) if cnt > 0 else 0.0

    sampled = df.sampleBy("attack_type", fractions=fractions, seed=args.seed)

    rank_window = Window.partitionBy("attack_type").orderBy(F.rand(args.seed))
    sampled = sampled.withColumn("row_rank", F.row_number().over(rank_window))
    target_map_expr = F.create_map([F.lit(x) for kv in TARGET_COUNTS.items() for x in kv])
    sampled = sampled.withColumn(
        "target_cap",
        F.coalesce(target_map_expr[F.col("attack_type")].cast("int"), F.lit(2147483647)),
    ).where(F.col("row_rank") <= F.col("target_cap")).drop("row_rank", "target_cap")

    final_cols = ["node_id", "event_time", "attack_type", "label_multiclass", "label_binary"]
    if "replica_id" in sampled.columns:
        final_cols.append("replica_id")
    final_cols.append("features")
    sampled = sampled.select(*final_cols)

    sampled.write.mode(args.output_mode).format(args.output_format).save(args.output)

    out_rows = validate_min_rows(sampled, args.min_output_rows, "Phase2B output")
    dist = class_distribution(sampled, "attack_type")

    write_metrics_json(
        args.metrics_output,
        {
            "phase": "phase2b_sample_nodes",
            "input_path": args.input,
            "output_path": args.output,
            "output_rows": out_rows,
            "distinct_classes": len(dist),
            "class_distribution": dist,
            "fractions": fractions,
            "spark_master": args.master,
            "shuffle_partitions": args.shuffle_partitions,
        },
    )

    print("Phase 2B complete")
    print(f"Output rows: {out_rows}")
    print(f"Output path: {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
