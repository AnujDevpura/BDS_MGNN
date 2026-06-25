import argparse

from pyspark.sql import functions as F

from pipeline_utils import build_spark, validate_min_rows, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 3C - heterogeneous graph fusion")
    p.add_argument("--temporal-input", default="artifacts/phase3_temporal")
    p.add_argument("--similarity-input", default="artifacts/phase3_similarity")
    p.add_argument("--output", default="artifacts/phase3_final")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase3c_metrics.json")
    p.add_argument("--temporal-weight-scale", type=float, default=1.0)
    p.add_argument("--similarity-weight-scale", type=float, default=0.25)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="8g")
    p.add_argument("--shuffle-partitions", type=int, default=96)
    p.add_argument("--default-parallelism", type=int, default=48)
    p.add_argument("--output-mode", default="overwrite")
    p.add_argument("--min-output-rows", type=int, default=200000)
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark(
        app_name="MGNN-Phase3C-Fusion",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    temporal = spark.read.parquet(args.temporal_input).withColumn(
        "weight", F.col("weight") * F.lit(args.temporal_weight_scale)
    )
    similarity = spark.read.parquet(args.similarity_input).withColumn(
        "weight", F.col("weight") * F.lit(args.similarity_weight_scale)
    )

    temporal = temporal.withColumn("edge_type_id", F.lit(0))
    similarity = similarity.withColumn("edge_type_id", F.lit(1))

    final_graph = (
        temporal.unionByName(similarity)
        .where(F.col("src") != F.col("dst"))
        .dropDuplicates(["src", "dst", "edge_type_id"])
        .repartition("edge_type_id")
    )

    final_graph.write.mode(args.output_mode).partitionBy("edge_type_id").parquet(args.output)
    edge_count = validate_min_rows(final_graph, args.min_output_rows, "Phase3C final graph")

    counts = {int(r["edge_type_id"]): int(r["count"]) for r in final_graph.groupBy("edge_type_id").count().collect()}
    write_metrics_json(
        args.metrics_output,
        {
            "phase": "phase3c_combine",
            "temporal_input": args.temporal_input,
            "similarity_input": args.similarity_input,
            "output_path": args.output,
            "total_edges": edge_count,
            "edge_type_counts": counts,
            "temporal_weight_scale": args.temporal_weight_scale,
            "similarity_weight_scale": args.similarity_weight_scale,
            "spark_master": args.master,
            "shuffle_partitions": args.shuffle_partitions,
        },
    )

    print("Phase 3C complete")
    print(f"Final edges: {edge_count}")
    spark.stop()


if __name__ == "__main__":
    main()
