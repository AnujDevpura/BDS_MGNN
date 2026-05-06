# phase3c_combine.py

# =========================================================
# MGNN PHASE 3C
# MULTI-RELATIONAL GRAPH FUSION
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================================================
# CONFIG
# =========================================================

# -----------------------------
# INPUTS
# -----------------------------

TEMPORAL_PATH = "artifacts/phase3_temporal"

SIMILARITY_PATH = "artifacts/phase3_similarity"

# -----------------------------
# OUTPUT
# -----------------------------

OUTPUT_PATH = "artifacts/phase3_final"

# -----------------------------
# GRAPH VERSIONING
# -----------------------------

GRAPH_NAME = "mgnn_multigraph"

EXPERIMENT_TAG = "550k_multiclass_final"

# -----------------------------
# CLEANUP
# -----------------------------

REMOVE_DUPLICATES = True

REMOVE_SELF_LOOPS = True

# -----------------------------
# EDGE REWEIGHTING
# -----------------------------

ENABLE_EDGE_REWEIGHTING = True

TEMPORAL_WEIGHT_SCALE = 1.0

SIMILARITY_WEIGHT_SCALE = 0.25

# -----------------------------
# RELATION IDS
# -----------------------------

TEMPORAL_RELATION_ID = 0

SIMILARITY_RELATION_ID = 1

# -----------------------------
# SPARK CONFIG
# -----------------------------

SPARK_DRIVER_MEMORY = "8g"

LOCAL_CORES = "local[12]"

SHUFFLE_PARTITIONS = 32

DEFAULT_PARALLELISM = 12

# -----------------------------
# STORAGE
# -----------------------------

OUTPUT_MODE = "overwrite"

SAVE_STATS = True

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
# LOAD TEMPORAL GRAPH
# =========================================================

print("\nLoading temporal graph...")

temporal = spark.read.parquet(
    TEMPORAL_PATH
)

temporal_edges = temporal.count()

print("Temporal edges:",
      temporal_edges)

# =========================================================
# LOAD SIMILARITY GRAPH
# =========================================================

print("\nLoading similarity graph...")

similarity = spark.read.parquet(
    SIMILARITY_PATH
)

similarity_edges = similarity.count()

print("Similarity edges:",
      similarity_edges)

# =========================================================
# OPTIONAL EDGE REWEIGHTING
# =========================================================

if ENABLE_EDGE_REWEIGHTING:

    print("\nApplying relation-specific weighting...")

    temporal = temporal.withColumn(

        "weight",

        F.col("weight")
        * TEMPORAL_WEIGHT_SCALE

    )

    similarity = similarity.withColumn(

        "weight",

        F.col("weight")
        * SIMILARITY_WEIGHT_SCALE

    )

# =========================================================
# RELATION ID ENCODING
# =========================================================

print("\nEncoding relation IDs...")

temporal = temporal.withColumn(

    "edge_type_id",

    F.lit(TEMPORAL_RELATION_ID)

)

similarity = similarity.withColumn(

    "edge_type_id",

    F.lit(SIMILARITY_RELATION_ID)

)

# =========================================================
# COMBINE HETEROGENEOUS GRAPH
# =========================================================

print("\nCombining multi-relational graph...")

final_graph = temporal.unionByName(
    similarity
)

# =========================================================
# REMOVE SELF LOOPS
# =========================================================

if REMOVE_SELF_LOOPS:

    print("\nRemoving self loops...")

    final_graph = final_graph.where(
        F.col("src") != F.col("dst")
    )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

if REMOVE_DUPLICATES:

    print("\nRemoving duplicate edges...")

    final_graph = final_graph.dropDuplicates(
        ["src", "dst", "edge_type"]
    )

# =========================================================
# CACHE FINAL GRAPH
# =========================================================

print("\nCaching final graph...")

final_graph = final_graph.cache()

# =========================================================
# SAVE GRAPH
# =========================================================

print("\nSaving final MGNN graph...")

final_graph.write.mode(
    OUTPUT_MODE
).parquet(
    OUTPUT_PATH
)

# =========================================================
# FINAL GRAPH STATISTICS
# =========================================================

if SAVE_STATS:

    print("\nComputing graph statistics...")

    total_edges = final_graph.count()

    unique_nodes = (

        final_graph

        .select("src")

        .union(

            final_graph.select(
                F.col("dst").alias("src")
            )

        )

        .distinct()

        .count()

    )

    avg_degree = total_edges / unique_nodes

    # -----------------------------------------------------
    # MAIN STATS
    # -----------------------------------------------------

    print("\n==============================")
    print("FINAL MGNN GRAPH STATS")
    print("==============================")

    print("\nGraph Name:",
          GRAPH_NAME)

    print("Experiment:",
          EXPERIMENT_TAG)

    print("\nUnique nodes:",
          unique_nodes)

    print("Total edges:",
          total_edges)

    print("\nAverage degree:",
          round(avg_degree, 2))

    # -----------------------------------------------------
    # EDGE TYPE COUNTS
    # -----------------------------------------------------

    print("\nEdge-type distribution:")

    final_graph.groupBy(
        "edge_type"
    ).count().show()

    # -----------------------------------------------------
    # RELATION ID COUNTS
    # -----------------------------------------------------

    print("\nRelation ID distribution:")

    final_graph.groupBy(
        "edge_type_id"
    ).count().show()

    # -----------------------------------------------------
    # WEIGHT STATS
    # -----------------------------------------------------

    print("\nGlobal weight statistics:")

    final_graph.describe(
        ["weight"]
    ).show()

    # -----------------------------------------------------
    # PER-RELATION WEIGHTS
    # -----------------------------------------------------

    print("\nPer-relation weight statistics:")

    final_graph.groupBy(
        "edge_type"
    ).agg(

        F.avg("weight").alias("avg_weight"),

        F.min("weight").alias("min_weight"),

        F.max("weight").alias("max_weight")

    ).show(
        truncate=False
    )

    # -----------------------------------------------------
    # SAMPLE EDGES
    # -----------------------------------------------------

    print("\nExample edges:")

    final_graph.show(
        10,
        truncate=False
    )

# =========================================================
# CLEANUP
# =========================================================

spark.stop()

print("\nPhase 3C graph fusion complete.")