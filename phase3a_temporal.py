import argparse

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pipeline_utils import build_spark, validate_min_rows, validate_null_rate, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 3A - temporal graph construction")
    p.add_argument("--input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--output", default="artifacts/phase3_temporal")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase3a_metrics.json")
    p.add_argument("--temporal-window", type=int, default=3)
    p.add_argument("--bucket-seconds", type=int, default=2)
    p.add_argument("--max-neighbors", type=int, default=8)
    p.add_argument("--min-edge-weight", type=float, default=0.2)
    p.add_argument("--allow-cross-bucket", action="store_true")
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="10g")
    p.add_argument("--shuffle-partitions", type=int, default=96)
    p.add_argument("--default-parallelism", type=int, default=48)
    p.add_argument("--output-mode", default="overwrite")
    p.add_argument("--min-output-rows", type=int, default=100000)
    p.add_argument("--max-null-time-rate", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark(
        app_name="MGNN-Phase3A-Temporal",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    df = spark.read.parquet(args.input).select("node_id", "event_time")
    input_rows = df.count()
    null_rate = validate_null_rate(df, "event_time", args.max_null_time_rate, total_rows=input_rows)
    df = df.where(F.col("event_time").isNotNull())

    df = df.withColumn("time_bucket", (F.unix_timestamp("event_time") / args.bucket_seconds).cast("long"))
    df = df.repartition("time_bucket")
    w = Window.partitionBy("time_bucket").orderBy("event_time")
    df = df.withColumn("row_num", F.row_number().over(w))

    left = df.alias("l")
    right = df.alias("r")
    if args.allow_cross_bucket:
        cond = (
            (F.abs(F.col("l.time_bucket") - F.col("r.time_bucket")) <= 1)
            & (F.col("l.row_num") < F.col("r.row_num"))
            & (F.col("r.row_num") <= F.col("l.row_num") + args.max_neighbors)
        )
    else:
        cond = (
            (F.col("l.time_bucket") == F.col("r.time_bucket"))
            & (F.col("l.row_num") < F.col("r.row_num"))
            & (F.col("r.row_num") <= F.col("l.row_num") + args.max_neighbors)
        )

    edges = left.join(right, cond, "inner").withColumn(
        "time_diff",
        F.abs(F.unix_timestamp(F.col("r.event_time")) - F.unix_timestamp(F.col("l.event_time"))),
    )
    edges = edges.where(F.col("time_diff") <= args.temporal_window)
    edges = edges.withColumn("weight", 1.0 / (1.0 + F.col("time_diff"))).where(F.col("weight") >= args.min_edge_weight)
    edges = (
        edges.select(F.col("l.node_id").alias("src"), F.col("r.node_id").alias("dst"), "weight")
        .where(F.col("src") != F.col("dst"))
        .dropDuplicates(["src", "dst"])
        .withColumn("edge_type", F.lit("temporal"))
    )

    edges.write.mode(args.output_mode).parquet(args.output)
    edge_count = validate_min_rows(edges, args.min_output_rows, "Phase3A temporal edges")

    write_metrics_json(
        args.metrics_output,
        {
            "phase": "phase3a_temporal",
            "input_path": args.input,
            "output_path": args.output,
            "input_rows": input_rows,
            "event_time_null_rate": null_rate,
            "edge_count": edge_count,
            "temporal_window": args.temporal_window,
            "bucket_seconds": args.bucket_seconds,
            "max_neighbors": args.max_neighbors,
            "spark_master": args.master,
            "shuffle_partitions": args.shuffle_partitions,
        },
    )
    print("Phase 3A complete")
    print(f"Temporal edges: {edge_count}")
    spark.stop()


if __name__ == "__main__":
    main()
