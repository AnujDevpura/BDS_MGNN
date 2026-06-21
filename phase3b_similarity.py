import argparse

from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pipeline_utils import build_spark, validate_min_rows, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 3B - distributed similarity graph")
    p.add_argument("--input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--output", default="artifacts/phase3_similarity")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase3b_metrics.json")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-similarity", type=float, default=0.80)
    p.add_argument("--bucket-length", type=float, default=2.0)
    p.add_argument("--num-hash-tables", type=int, default=3)
    p.add_argument("--max-distance", type=float, default=0.65)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="10g")
    p.add_argument("--shuffle-partitions", type=int, default=96)
    p.add_argument("--default-parallelism", type=int, default=48)
    p.add_argument("--output-mode", default="overwrite")
    p.add_argument("--min-output-rows", type=int, default=100000)
    p.add_argument("--max-nodes", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark(
        app_name="MGNN-Phase3B-SimilarityLSH",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    df = spark.read.parquet(args.input).select("node_id", "features")
    if "features" not in df.columns:
        raise ValueError("Missing required features column")
    if args.max_nodes > 0:
        df = df.orderBy("node_id").limit(args.max_nodes)

    normalized = Normalizer(inputCol="features", outputCol="features_l2", p=2.0).transform(df)

    lsh = BucketedRandomProjectionLSH(
        inputCol="features_l2",
        outputCol="hashes",
        bucketLength=args.bucket_length,
        numHashTables=args.num_hash_tables,
    )
    model = lsh.fit(normalized)

    approx = model.approxSimilarityJoin(
        normalized.alias("a"), normalized.alias("b"), args.max_distance, distCol="dist"
    )
    pairs = (
        approx.where(F.col("datasetA.node_id") != F.col("datasetB.node_id"))
        .select(
            F.col("datasetA.node_id").alias("src"),
            F.col("datasetB.node_id").alias("dst"),
            F.col("dist").alias("dist"),
        )
        .dropDuplicates(["src", "dst"])
    )

    # For L2-normalized vectors: cosine_similarity = 1 - squared_euclidean_distance / 2.
    pairs = pairs.withColumn(
        "weight",
        F.greatest(F.lit(-1.0), F.least(F.lit(1.0), F.lit(1.0) - (F.col("dist") ** 2) / 2.0)),
    ).where(F.col("weight") >= F.lit(args.min_similarity))

    knn_window = Window.partitionBy("src").orderBy(F.col("weight").desc())
    edges = (
        pairs.withColumn("rnk", F.row_number().over(knn_window))
        .where(F.col("rnk") <= args.top_k)
        .drop("rnk", "dist")
        .withColumn("edge_type", F.lit("similarity"))
    )

    edges.write.mode(args.output_mode).parquet(args.output)
    edge_count = validate_min_rows(edges, args.min_output_rows, "Phase3B similarity edges")

    write_metrics_json(
        args.metrics_output,
        {
            "phase": "phase3b_similarity",
            "input_path": args.input,
            "output_path": args.output,
            "edge_count": edge_count,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "bucket_length": args.bucket_length,
            "num_hash_tables": args.num_hash_tables,
            "max_distance": args.max_distance,
            "spark_master": args.master,
            "shuffle_partitions": args.shuffle_partitions,
        },
    )

    print("Phase 3B complete")
    print(f"Similarity edges: {edge_count}")
    spark.stop()


if __name__ == "__main__":
    main()
