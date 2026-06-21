import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import LongType, StructField, StructType

from pipeline_utils import build_spark, validate_min_rows, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 4 - stream-safe PyG export")
    p.add_argument("--feature-input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--edge-input", default="artifacts/phase3_final")
    p.add_argument("--output-dir", default="artifacts/phase4_pyg")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase4_metrics.json")
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="10g")
    p.add_argument("--shuffle-partitions", type=int, default=96)
    p.add_argument("--default-parallelism", type=int, default=48)
    p.add_argument("--min-nodes", type=int, default=50000)
    p.add_argument("--min-edges", type=int, default=100000)
    p.add_argument("--sampling-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark(
        app_name="MGNN-Phase4-ExportPyG",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        default_parallelism=args.default_parallelism,
        enable_adaptive=True,
        enable_dynamic_allocation=False,
        parquet_compression="zstd",
    )

    feature_df = spark.read.parquet(args.feature_input).select("node_id", "features", "attack_type", "label_multiclass")
    if args.max_nodes > 0:
        counts = feature_df.groupBy("label_multiclass").count().collect()
        total = sum(int(row["count"]) for row in counts)
        quotas = {
            int(row["label_multiclass"]): max(
                1, min(int(row["count"]), round(args.max_nodes * int(row["count"]) / total))
            )
            for row in counts
        }
        quota_expr = F.create_map([F.lit(x) for item in quotas.items() for x in item])
        sample_window = Window.partitionBy("label_multiclass").orderBy(F.rand(args.sampling_seed))
        feature_df = (
            feature_df.withColumn("sample_rank", F.row_number().over(sample_window))
            .where(F.col("sample_rank") <= quota_expr[F.col("label_multiclass")])
            .drop("sample_rank")
        )
    feature_df = feature_df.cache()

    edge_df = spark.read.parquet(args.edge_input).select("src", "dst", "weight", "edge_type_id")

    node_rdd = feature_df.select("node_id").distinct().orderBy("node_id").rdd.zipWithIndex()
    schema = StructType([StructField("node_id", LongType(), False), StructField("node_idx", LongType(), False)])
    node_map_df = spark.createDataFrame(node_rdd.map(lambda x: (x[0]["node_id"], x[1])), schema=schema).cache()

    mapped_features = (
        feature_df.join(node_map_df, on="node_id", how="inner")
        .withColumn("features_array", vector_to_array("features"))
        .select("node_idx", "features_array", "label_multiclass", "attack_type")
        .orderBy("node_idx")
    )

    valid_nodes = node_map_df.select("node_id").cache()
    mapped_edges = (
        edge_df.join(node_map_df.withColumnRenamed("node_id", "src").withColumnRenamed("node_idx", "src_idx"), on="src", how="inner")
        .join(node_map_df.withColumnRenamed("node_id", "dst").withColumnRenamed("node_idx", "dst_idx"), on="dst", how="inner")
        .select("src_idx", "dst_idx", "weight", "edge_type_id")
    )

    num_nodes = validate_min_rows(mapped_features, args.min_nodes, "Phase4 mapped nodes")
    num_edges = validate_min_rows(mapped_edges, args.min_edges, "Phase4 mapped edges")

    first_row = mapped_features.select("features_array").first()
    feat_dim = len(first_row["features_array"])

    x_np = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    y_np = np.zeros((num_nodes,), dtype=np.int64)
    attack_types = [None] * num_nodes

    for row in mapped_features.toLocalIterator():
        idx = int(row["node_idx"])
        x_np[idx, :] = np.asarray(row["features_array"], dtype=np.float32)
        y_np[idx] = int(row["label_multiclass"])
        attack_types[idx] = row["attack_type"]

    edge_index_np = np.zeros((2, num_edges), dtype=np.int64)
    edge_weight_np = np.zeros((num_edges,), dtype=np.float32)
    edge_type_np = np.zeros((num_edges,), dtype=np.int64)

    e = 0
    for row in mapped_edges.toLocalIterator():
        edge_index_np[0, e] = int(row["src_idx"])
        edge_index_np[1, e] = int(row["dst_idx"])
        edge_weight_np[e] = float(row["weight"])
        edge_type_np[e] = int(row["edge_type_id"])
        e += 1

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    edge_index = torch.from_numpy(edge_index_np)
    edge_weight = torch.from_numpy(edge_weight_np)
    edge_type = torch.from_numpy(edge_type_np)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x, output_dir / "x.pt")
    torch.save(y, output_dir / "y.pt")
    torch.save(edge_index, output_dir / "edge_index.pt")
    torch.save(edge_weight, output_dir / "edge_weight.pt")
    torch.save(edge_type, output_dir / "edge_type.pt")
    torch.save(attack_types, output_dir / "attack_types.pt")

    node_map_pd = pd.DataFrame({"node_id": [r["node_id"] for r in node_map_df.orderBy("node_idx").toLocalIterator()]})
    node_map_pd["node_idx"] = np.arange(len(node_map_pd), dtype=np.int64)
    node_map_pd.to_parquet(output_dir / "node_mapping.parquet", index=False)

    write_metrics_json(
        args.metrics_output,
        {
            "phase": "phase4_export_pyg",
            "feature_input": args.feature_input,
            "edge_input": args.edge_input,
            "output_dir": str(output_dir),
            "nodes": int(num_nodes),
            "edges": int(num_edges),
            "feature_dim": int(feat_dim),
            "temporal_edges": int((edge_type_np == 0).sum()),
            "similarity_edges": int((edge_type_np == 1).sum()),
            "spark_master": args.master,
            "shuffle_partitions": args.shuffle_partitions,
        },
    )

    print("Phase 4 export complete")
    print(f"x shape: {x.shape} | y shape: {y.shape}")
    print(f"edge_index shape: {edge_index.shape}")
    spark.stop()


if __name__ == "__main__":
    main()
