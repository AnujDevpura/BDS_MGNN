# phase3a_temporal.py

# =========================================================
# MGNN PHASE 3A
# DISTRIBUTED TEMPORAL GRAPH CONSTRUCTION
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================================================
# CONFIG
# =========================================================

# -----------------------------
# INPUT / OUTPUT
# -----------------------------

INPUT_PATH = "artifacts/phase2_sampled_500k"

OUTPUT_PATH = "artifacts/phase3_temporal"

# -----------------------------
# GRAPH VERSIONING
# -----------------------------

GRAPH_NAME = "temporal_graph"

EXPERIMENT_TAG = "temporal_k8_w3_550k"

# -----------------------------
# TEMPORAL GRAPH
# -----------------------------

TEMPORAL_WINDOW = 3

BUCKET_SECONDS = 2

MAX_TEMP_NEIGHBORS = 8

ALLOW_CROSS_BUCKET = False

# -----------------------------
# EDGE FILTERING
# -----------------------------

MIN_EDGE_WEIGHT = 0.2

REMOVE_DUPLICATES = True

REMOVE_SELF_LOOPS = True

# -----------------------------
# SPARK CONFIG
# -----------------------------

SPARK_DRIVER_MEMORY = "10g"

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
# LOAD DATA
# =========================================================

print("\nLoading Phase 2 sampled nodes...")

df = spark.read.parquet(INPUT_PATH).select(

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

)

initial_rows = df.count()

print("Initial rows:", initial_rows)

# =========================================================
# VALIDATE TIMESTAMPS
# =========================================================

null_timestamps = (

    df.where(
        F.col("event_time").isNull()
    ).count()

)

print("Null timestamps:", null_timestamps)

if null_timestamps > 0:

    print("\nDropping rows with null timestamps...")

    df = df.where(
        F.col("event_time").isNotNull()
    )

# =========================================================
# CREATE TIME BUCKETS
# =========================================================

print("\nCreating temporal buckets...")

df = df.withColumn(

    "time_bucket",

    (
        F.unix_timestamp("event_time")
        / BUCKET_SECONDS
    ).cast("long")

)

# =========================================================
# REPARTITION FOR DISTRIBUTION
# =========================================================

print("\nRepartitioning by time bucket...")

df = df.repartition("time_bucket")

# =========================================================
# ORDER EVENTS INSIDE BUCKETS
# =========================================================

print("\nOrdering events inside buckets...")

window_spec = (

    Window.partitionBy(
        "time_bucket"
    )

    .orderBy(
        "event_time"
    )

)

df = df.withColumn(

    "row_num",

    F.row_number().over(window_spec)

)

# =========================================================
# BUILD TEMPORAL EDGES
# =========================================================

print("\nConstructing temporal edges...")

left = df.alias("l")

right = df.alias("r")

# ---------------------------------------------------------
# SAME BUCKET TEMPORAL EDGES
# ---------------------------------------------------------

join_condition = (

    (F.col("l.time_bucket")
     ==
     F.col("r.time_bucket"))

    &

    (F.col("l.row_num")
     <
     F.col("r.row_num"))

    &

    (
        F.col("r.row_num")
        <=
        F.col("l.row_num")
        + MAX_TEMP_NEIGHBORS
    )

)

# ---------------------------------------------------------
# OPTIONAL ADJACENT BUCKET SUPPORT
# ---------------------------------------------------------

if ALLOW_CROSS_BUCKET:

    join_condition = (

        (
            F.abs(
                F.col("l.time_bucket")
                -
                F.col("r.time_bucket")
            ) <= 1
        )

        &

        (F.col("l.row_num")
         <
         F.col("r.row_num"))

        &

        (
            F.col("r.row_num")
            <=
            F.col("l.row_num")
            + MAX_TEMP_NEIGHBORS
        )

    )

edges = left.join(

    right,

    join_condition,

    "inner"

)

# =========================================================
# EXACT TEMPORAL FILTER
# =========================================================

print("\nApplying exact temporal filtering...")

edges = edges.withColumn(

    "time_diff",

    F.abs(

        F.unix_timestamp(
            F.col("r.event_time")
        )

        -

        F.unix_timestamp(
            F.col("l.event_time")
        )

    )

)

edges = edges.where(
    F.col("time_diff") <= TEMPORAL_WINDOW
)

# =========================================================
# EDGE WEIGHTS
# =========================================================

print("\nComputing edge weights...")

edges = edges.withColumn(

    "weight",

    1.0 / (
        1.0 + F.col("time_diff")
    )

)

# =========================================================
# EDGE FILTERING
# =========================================================

edges = edges.where(
    F.col("weight") >= MIN_EDGE_WEIGHT
)

# =========================================================
# FINAL EDGE FORMAT
# =========================================================

edges = edges.select(

    F.col("l.node_id").alias("src"),

    F.col("r.node_id").alias("dst"),

    "weight",

).withColumn(

    "edge_type",

    F.lit("temporal")

)

# =========================================================
# REMOVE SELF LOOPS
# =========================================================

if REMOVE_SELF_LOOPS:

    print("\nRemoving self loops...")

    edges = edges.where(
        F.col("src") != F.col("dst")
    )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

if REMOVE_DUPLICATES:

    print("\nRemoving duplicate edges...")

    edges = edges.dropDuplicates(
        ["src", "dst"]
    )

# =========================================================
# SAVE GRAPH
# =========================================================

print("\nSaving temporal graph...")

edges.write.mode(
    OUTPUT_MODE
).parquet(
    OUTPUT_PATH
)

# =========================================================
# GRAPH STATISTICS
# =========================================================

if SAVE_STATS:

    print("\nComputing graph statistics...")

    edge_count = edges.count()

    final_nodes = df.count()

    avg_degree = edge_count / final_nodes

    print("\n==============================")
    print("TEMPORAL GRAPH STATS")
    print("==============================")

    print("\nGraph Name:",
          GRAPH_NAME)

    print("Experiment:",
          EXPERIMENT_TAG)

    print("\nNodes:",
          final_nodes)

    print("Edges:",
          edge_count)

    print("\nAverage degree:",
          round(avg_degree, 2))

    print("\nTemporal window:",
          TEMPORAL_WINDOW)

    print("Bucket seconds:",
          BUCKET_SECONDS)

    print("Max temporal neighbors:",
          MAX_TEMP_NEIGHBORS)

    # -----------------------------------------------------
    # CLASS COVERAGE
    # -----------------------------------------------------

    print("\nTemporal node coverage by class:")

    class_stats = (

        df.groupBy(
            "attack_type"
        )

        .count()

        .orderBy(
            F.desc("count")
        )

    )

    class_stats.show(
        50,
        truncate=False
    )

    # -----------------------------------------------------
    # SAMPLE EDGES
    # -----------------------------------------------------

    print("\nExample edges:")

    edges.show(
        10,
        truncate=False
    )

# =========================================================
# CLEANUP
# =========================================================

spark.stop()

print("\nPhase 3A temporal graph complete.")