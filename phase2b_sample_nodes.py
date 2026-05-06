# phase2b_sample_nodes.py

# =========================================================
# MGNN PHASE 2B
# STRATIFIED CANONICAL NODE UNIVERSE
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================================================
# CONFIG
# =========================================================

# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------

INPUT_PATH = "artifacts/phase2_features"

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

OUTPUT_PATH = "artifacts/phase2_sampled_500k"

# ---------------------------------------------------------
# TARGET CLASS COUNTS
# ---------------------------------------------------------

TARGET_COUNTS = {

    # major classes
    "benign": 150000,

    "dos_hulk": 100000,

    "ddos": 80000,

    "portscan": 80000,

    # medium classes
    "dos_goldeneye": 30000,

    "ftp_patator": 25000,

    "ssh_patator": 25000,

    "dos_slowloris": 20000,

    "dos_slowhttptest": 20000,

    # smaller classes
    "bot": 9000,

    "web_bruteforce": 7000,

    "web_xss": 3000,

    # preserve all rare classes
    "infiltration": 1000000,

    "web_sql_injection": 1000000,

    "heartbleed": 1000000,

}

# ---------------------------------------------------------
# RANDOMNESS
# ---------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------
# SPARK
# ---------------------------------------------------------

SPARK_DRIVER_MEMORY = "8g"

LOCAL_CORES = "local[12]"

SHUFFLE_PARTITIONS = 32

DEFAULT_PARALLELISM = 12

# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

OUTPUT_MODE = "overwrite"

# =========================================================
# SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName("MGNN-Phase2B-StratifiedSampling")
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
# LOAD PHASE 2 FEATURES
# =========================================================

print("\nLoading Phase 2 feature dataset...")

df = spark.read.parquet(INPUT_PATH)

total_rows = df.count()

print("Total rows:", total_rows)

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_cols = [

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

    "label_binary",

    "features",

]

missing = [

    c for c in required_cols
    if c not in df.columns

]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )

print("\nRequired columns verified")

# =========================================================
# REMOVE DUPLICATE NODE IDS
# =========================================================

print("\nRemoving duplicate node IDs...")

df = df.dropDuplicates(["node_id"])

unique_nodes = df.count()

print("Unique nodes:", unique_nodes)

# =========================================================
# ORIGINAL CLASS DISTRIBUTION
# =========================================================

print("\nOriginal class distribution:")

class_counts = (

    df.groupBy("attack_type")
    .count()

)

class_counts.orderBy(
    F.desc("count")
).show(
    50,
    truncate=False
)

# =========================================================
# STRATIFIED SAMPLING
# =========================================================

print("\nPerforming stratified sampling...")

sampled_parts = []

for attack_class, target_count in TARGET_COUNTS.items():

    print(f"\nProcessing class: {attack_class}")

    class_df = df.filter(
        F.col("attack_type") == attack_class
    )

    actual_count = class_df.count()

    print("Actual rows:", actual_count)

    # -----------------------------------------------------
    # KEEP ALL SMALL CLASSES
    # -----------------------------------------------------

    if actual_count <= target_count:

        print("Keeping all rows")

        sampled_parts.append(class_df)

    # -----------------------------------------------------
    # DOWNSAMPLE LARGE CLASSES
    # -----------------------------------------------------

    else:

        print(f"Sampling down to {target_count}")

        sampled_class = (

            class_df

            .sample(
                withReplacement=False,
                fraction=target_count / actual_count,
                seed=RANDOM_SEED
            )
            .limit(target_count)

        )

        sampled_parts.append(sampled_class)

# =========================================================
# UNION ALL SAMPLED CLASSES
# =========================================================

print("\nCombining sampled classes...")

sampled_df = sampled_parts[0]

for extra in sampled_parts[1:]:

    sampled_df = sampled_df.unionByName(extra)

# =========================================================
# FINAL COLUMN ORDER
# =========================================================

final_cols = [

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

    "label_binary",

]

if "replica_id" in sampled_df.columns:

    final_cols.append("replica_id")

final_cols.append("features")

sampled_df = sampled_df.select(*final_cols)

# =========================================================
# SAVE
# =========================================================

print("\nSaving canonical node universe...")

sampled_df.write.mode(
    OUTPUT_MODE
).parquet(
    OUTPUT_PATH
)

# =========================================================
# VALIDATION
# =========================================================

print("\n==============================")
print("PHASE 2B COMPLETE")
print("==============================")

final_count = sampled_df.count()

print("\nTotal sampled nodes:",
      final_count)

print("\nOutput path:")
print(OUTPUT_PATH)

print("\nFinal class distribution:")

sampled_df.groupBy(
    "attack_type"
).count().orderBy(
    F.desc("count")
).show(
    50,
    truncate=False
)

print("\nFinal multiclass distribution:")

sampled_df.groupBy(
    "label_multiclass"
).count().orderBy(
    "label_multiclass"
).show(
    50,
    truncate=False
)

print("\nSchema:")

sampled_df.printSchema()

print("\nSample rows:")

sampled_df.show(
    5,
    truncate=False
)

# =========================================================
# CLEANUP
# =========================================================

spark.stop()

print("\nCanonical stratified node universe created.")