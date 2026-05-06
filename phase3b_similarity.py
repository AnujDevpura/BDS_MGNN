# phase3b_similarity.py

# =========================================================
# MGNN PHASE 3B
# FAISS IVF COSINE SIMILARITY GRAPH
# =========================================================

import gc
import faiss
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array

# =========================================================
# CONFIG
# =========================================================

# -----------------------------
# INPUT / OUTPUT
# -----------------------------

INPUT_PATH = "artifacts/phase2_sampled_500k"

OUTPUT_PATH = "artifacts/phase3_similarity"

# -----------------------------
# GRAPH VERSIONING
# -----------------------------

GRAPH_NAME = "similarity_graph"

EXPERIMENT_TAG = "ivf_k8_550k"

# -----------------------------
# SIMILARITY GRAPH
# -----------------------------

TOP_K = 3

MIN_SIMILARITY = 0.80

REMOVE_DUPLICATES = True

REMOVE_SELF_LOOPS = True

# -----------------------------
# FAISS IVF
# -----------------------------

NLIST = 256

NPROBE = 16

FAISS_THREADS = 6

# -----------------------------
# SPARK CONFIG
# -----------------------------

SPARK_DRIVER_MEMORY = "10g"

LOCAL_CORES = "local[12]"

# -----------------------------
# STORAGE
# -----------------------------

OUTPUT_MODE = "overwrite"

SAVE_STATS = True

# =========================================================
# FAISS THREADING
# =========================================================

faiss.omp_set_num_threads(FAISS_THREADS)

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
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark started")

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading Phase 2 sampled features...")

df = (
    spark.read.parquet(INPUT_PATH)
    .select(
        "node_id",
        "attack_type",
        "label_multiclass",
        "features"
    )
)

node_count = df.count()

print("Loaded nodes:", node_count)

# =========================================================
# VECTOR -> ARRAY
# =========================================================

print("\nConverting Spark vectors...")

df = df.withColumn(
    "features_array",
    vector_to_array("features")
)

# =========================================================
# COLLECT TO PANDAS
# =========================================================

print("\nCollecting vectors to NumPy...")

pdf = df.select(
    "node_id",
    "features_array"
).toPandas()

node_ids = pdf["node_id"].values.astype(np.int64)

vectors = np.vstack(
    pdf["features_array"].values
).astype("float32")

print("Vector matrix shape:", vectors.shape)

# =========================================================
# CLEAN VECTOR VALUES
# =========================================================

print("\nCleaning vectors...")

vectors = np.nan_to_num(
    vectors,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

vectors = np.clip(
    vectors,
    -1e6,
    1e6
)

print("NaN count:",
      np.isnan(vectors).sum())

print("Inf count:",
      np.isinf(vectors).sum())

# =========================================================
# FREE SPARK MEMORY
# =========================================================

spark.stop()

print("\nSpark stopped to free RAM")

gc.collect()

# =========================================================
# NORMALIZE FOR COSINE SIMILARITY
# =========================================================

print("\nNormalizing vectors...")

faiss.normalize_L2(vectors)

# =========================================================
# BUILD IVF INDEX
# =========================================================

dimension = vectors.shape[1]

print("\nBuilding IVF index...")

quantizer = faiss.IndexFlatIP(dimension)

index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    NLIST,
    faiss.METRIC_INNER_PRODUCT
)

# =========================================================
# TRAIN INDEX
# =========================================================

print("\nTraining IVF index...")

index.train(vectors)

print("IVF index trained")

# =========================================================
# ADD VECTORS
# =========================================================

print("\nAdding vectors to index...")

index.add(vectors)

print("Indexed vectors:", index.ntotal)

# =========================================================
# SEARCH CONFIG
# =========================================================

index.nprobe = NPROBE

# =========================================================
# ANN SEARCH
# =========================================================

print("\nRunning ANN similarity search...")
print("This may take several minutes...")

distances, indices = index.search(
    vectors,
    TOP_K + 1
)

print("ANN search complete")

# =========================================================
# BUILD EDGE LIST
# =========================================================

print("\nBuilding similarity edges...")

edges = []

num_nodes = len(node_ids)

for i in range(num_nodes):

    src = int(node_ids[i])

    for j in range(1, TOP_K + 1):

        neighbor_idx = int(indices[i][j])

        # invalid neighbor
        if neighbor_idx < 0:
            continue

        dst = int(node_ids[neighbor_idx])

        similarity = float(distances[i][j])

        # weak similarity
        if similarity < MIN_SIMILARITY:
            continue

        # self loop removal
        if REMOVE_SELF_LOOPS and src == dst:
            continue

        edges.append(
            (
                src,
                dst,
                similarity,
                "similarity"
            )
        )

    if i % 10000 == 0:
        print(f"Processed {i}/{num_nodes}")

# =========================================================
# CREATE EDGE DATAFRAME
# =========================================================

print("\nCreating edge dataframe...")

edge_df = pd.DataFrame(
    edges,
    columns=[
        "src",
        "dst",
        "weight",
        "edge_type"
    ]
)

print("Raw similarity edges:",
      len(edge_df))

# =========================================================
# REMOVE DUPLICATES
# =========================================================

if REMOVE_DUPLICATES:

    edge_df = edge_df.drop_duplicates(
        subset=["src", "dst"]
    )

print("Unique similarity edges:",
      len(edge_df))

# =========================================================
# SAVE GRAPH
# =========================================================

print("\nSaving similarity graph...")

edge_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Similarity graph saved")

# =========================================================
# FINAL STATS
# =========================================================

if SAVE_STATS:

    avg_degree = len(edge_df) / num_nodes

    print("\n==============================")
    print("SIMILARITY GRAPH STATS")
    print("==============================")

    print("Graph Name:",
          GRAPH_NAME)

    print("Experiment:",
          EXPERIMENT_TAG)

    print("\nNodes:",
          num_nodes)

    print("Edges:",
          len(edge_df))

    print("\nAverage degree:",
          round(avg_degree, 2))

    print("\nTop-K neighbors:",
          TOP_K)

    print("Minimum similarity:",
          MIN_SIMILARITY)

    print("\nSimilarity weight statistics:")

    print(
        edge_df["weight"].describe()
    )

    print("\nExample edges:")

    print(
        edge_df.head(10)
    )

print("\nPhase 3B similarity graph complete.")