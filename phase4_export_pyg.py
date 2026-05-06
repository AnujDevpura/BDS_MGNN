# phase4_export_pyg.py

# =========================================================
# MGNN PHASE 4
# EXPORT MULTI-RELATIONAL GRAPH TO PYTORCH GEOMETRIC
# =========================================================

import gc
from pathlib import Path

import pandas as pd
import torch

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

# =========================================================
# CONFIG
# =========================================================

# -----------------------------
# INPUTS
# -----------------------------

FEATURE_INPUT = "artifacts/phase2_sampled_500k"

EDGE_INPUT = "artifacts/phase3_final"

# -----------------------------
# OUTPUT
# -----------------------------

OUTPUT_DIR = "artifacts/phase4_pyg"

# -----------------------------
# GRAPH VERSIONING
# -----------------------------

GRAPH_NAME = "mgnn_pyg_graph"

EXPERIMENT_TAG = "550k_multiclass"

# -----------------------------
# OPTIONAL NODE LIMIT
# -----------------------------

MAX_NODES = 0

# -----------------------------
# SPARK CONFIG
# -----------------------------

SPARK_DRIVER_MEMORY = "10g"

LOCAL_CORES = "local[12]"

SHUFFLE_PARTITIONS = 32

DEFAULT_PARALLELISM = 12

# =========================================================
# SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName(f"{GRAPH_NAME}_{EXPERIMENT_TAG}")
    .master(LOCAL_CORES)
    .config(
        "spark.driver.memory",
        SPARK_DRIVER_MEMORY
    )
    .config(
        "spark.sql.shuffle.partitions",
        SHUFFLE_PARTITIONS
    )
    .config(
        "spark.default.parallelism",
        DEFAULT_PARALLELISM
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark started")

# =========================================================
# LOAD FEATURE DATA
# =========================================================

print("\nLoading node features...")

feature_df = spark.read.parquet(
    FEATURE_INPUT
).select(

    "node_id",

    "features",

    "attack_type",

    "label_multiclass"

)

# =========================================================
# OPTIONAL NODE LIMIT
# =========================================================

if MAX_NODES > 0:

    print(f"\nLimiting to {MAX_NODES} nodes...")

    feature_df = feature_df.limit(MAX_NODES)

# =========================================================
# LOAD EDGE GRAPH
# =========================================================

print("\nLoading final graph...")

edge_df = spark.read.parquet(
    EDGE_INPUT
).select(

    "src",

    "dst",

    "weight",

    "edge_type",

    "edge_type_id"

)

# =========================================================
# VALID NODE SET
# =========================================================

print("\nBuilding valid node set...")

valid_nodes = feature_df.select(
    "node_id"
).distinct()

# =========================================================
# FILTER INVALID EDGES
# =========================================================

print("\nFiltering edges...")

edge_df = (

    edge_df

    .join(

        valid_nodes.withColumnRenamed(
            "node_id",
            "src"
        ),

        on="src",

        how="inner"

    )

    .join(

        valid_nodes.withColumnRenamed(
            "node_id",
            "dst"
        ),

        on="dst",

        how="inner"

    )

)

# =========================================================
# CREATE DETERMINISTIC NODE MAPPING
# =========================================================

print("\nCreating contiguous node mapping...")

node_df = (

    feature_df

    .select("node_id")

    .distinct()

    .orderBy("node_id")

)

node_rows = node_df.collect()

node_to_idx = {

    row["node_id"]: idx

    for idx, row in enumerate(node_rows)

}

print("Mapped nodes:",
      len(node_to_idx))

# =========================================================
# CONVERT FEATURES TO ARRAYS
# =========================================================

print("\nConverting feature vectors...")

feature_df = feature_df.withColumn(

    "features_array",

    vector_to_array("features")

)

# =========================================================
# COLLECT FEATURES
# =========================================================

print("\nCollecting node features...")

feature_pdf = feature_df.select(

    "node_id",

    "features_array",

    "attack_type",

    "label_multiclass"

).toPandas()

# =========================================================
# MAP NODE INDICES
# =========================================================

feature_pdf["node_idx"] = (

    feature_pdf["node_id"]

    .map(node_to_idx)

)

feature_pdf = feature_pdf.sort_values(
    "node_idx"
)

# =========================================================
# BUILD FEATURE TENSOR
# =========================================================

print("\nBuilding feature tensor...")

x = torch.tensor(

    feature_pdf["features_array"].tolist(),

    dtype=torch.float32

)

# =========================================================
# BUILD LABEL TENSOR
# =========================================================

print("\nBuilding multiclass label tensor...")

y = torch.tensor(

    feature_pdf["label_multiclass"]
    .astype(int)
    .tolist(),

    dtype=torch.long

)

print("Feature tensor shape:",
      x.shape)

print("Label tensor shape:",
      y.shape)

# =========================================================
# EXPORT ATTACK TYPES
# =========================================================

print("\nSaving attack type metadata...")

attack_types = feature_pdf[
    "attack_type"
].tolist()

# =========================================================
# COLLECT EDGES
# =========================================================

print("\nCollecting edges...")

edge_pdf = edge_df.toPandas()

print("Edges collected:",
      len(edge_pdf))

# =========================================================
# MAP EDGE NODE IDS
# =========================================================

print("\nMapping edge indices...")

edge_pdf["src_idx"] = (
    edge_pdf["src"]
    .map(node_to_idx)
)

edge_pdf["dst_idx"] = (
    edge_pdf["dst"]
    .map(node_to_idx)
)

# =========================================================
# REMOVE INVALID EDGES
# =========================================================

edge_pdf = edge_pdf.dropna(
    subset=["src_idx", "dst_idx"]
)

print("Valid edges:",
      len(edge_pdf))

# =========================================================
# BUILD EDGE INDEX
# =========================================================

print("\nBuilding edge index tensor...")

edge_index = torch.tensor(

    [

        edge_pdf["src_idx"]
        .astype(int)
        .tolist(),

        edge_pdf["dst_idx"]
        .astype(int)
        .tolist()

    ],

    dtype=torch.long

)

# =========================================================
# BUILD EDGE WEIGHT TENSOR
# =========================================================

print("\nBuilding edge weight tensor...")

edge_weight = torch.tensor(

    edge_pdf["weight"]
    .astype(float)
    .tolist(),

    dtype=torch.float32

)

# =========================================================
# BUILD EDGE TYPE TENSOR
# =========================================================

print("\nBuilding edge type tensor...")

edge_type = torch.tensor(

    edge_pdf["edge_type_id"]
    .astype(int)
    .tolist(),

    dtype=torch.long

)

# =========================================================
# CLEANUP SPARK
# =========================================================

spark.stop()

gc.collect()

print("\nSpark stopped")

# =========================================================
# CREATE OUTPUT DIR
# =========================================================

output_dir = Path(OUTPUT_DIR)

output_dir.mkdir(

    parents=True,

    exist_ok=True

)

# =========================================================
# SAVE TENSORS
# =========================================================

print("\nSaving PyG tensors...")

torch.save(
    x,
    output_dir / "x.pt"
)

torch.save(
    y,
    output_dir / "y.pt"
)

torch.save(
    edge_index,
    output_dir / "edge_index.pt"
)

torch.save(
    edge_weight,
    output_dir / "edge_weight.pt"
)

torch.save(
    edge_type,
    output_dir / "edge_type.pt"
)

torch.save(
    attack_types,
    output_dir / "attack_types.pt"
)

# =========================================================
# SAVE NODE MAPPING
# =========================================================

print("\nSaving node mapping...")

mapping_df = pd.DataFrame({

    "node_id": list(node_to_idx.keys()),

    "node_idx": list(node_to_idx.values())

})

mapping_df.to_parquet(

    output_dir / "node_mapping.parquet",

    index=False

)

# =========================================================
# FINAL VALIDATION
# =========================================================

print("\n==============================")
print("PHASE 4 EXPORT COMPLETE")
print("==============================")

print("\nGraph Name:",
      GRAPH_NAME)

print("Experiment:",
      EXPERIMENT_TAG)

print("\nNode feature tensor:",
      x.shape)

print("Label tensor:",
      y.shape)

print("\nEdge index tensor:",
      edge_index.shape)

print("Edge weight tensor:",
      edge_weight.shape)

print("Edge type tensor:",
      edge_type.shape)

print("\nTemporal edges:",
      (edge_type == 0).sum().item())

print("Similarity edges:",
      (edge_type == 1).sum().item())

# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\nMulticlass label distribution:")

print(

    feature_pdf["label_multiclass"]

    .value_counts()

    .sort_index()

)

print("\nSaved to:",
      output_dir)

print("\nPhase 4 complete.")