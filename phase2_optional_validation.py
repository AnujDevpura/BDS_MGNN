# phase2_optional_validation.py

# =========================================================
# MGNN PHASE 2 VALIDATION GATE
# VALIDATE FINAL FEATURE VECTORS
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.functions import vector_to_array

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = "artifacts/phase2_features"

OUTPUT_CSV_PATH = "artifacts/phase2_validation_csv"

# =========================================================
# SPARK
# =========================================================

spark = (
    SparkSession.builder
    .appName("MGNN-Phase2-Validation")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# =========================================================
# LOAD
# =========================================================

print("=== Phase 2 POST-CLEAN Validation Gate ===")

df = spark.read.parquet(INPUT_PATH)

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_cols = [

    "node_id",

    "attack_type",

    "label_multiclass",

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

# =========================================================
# VECTOR TO ARRAY
# =========================================================

df = df.withColumn(
    "features_array",
    vector_to_array("features")
)

# =========================================================
# BASIC STATS
# =========================================================

row_count = df.count()

vector_length = len(

    df.select("features_array")
    .first()["features_array"]

)

total_values = row_count * vector_length

# =========================================================
# NAN COUNTS
# =========================================================

nan_exprs = [

    F.when(
        F.isnan(F.col("features_array")[i]),
        1
    ).otherwise(0)

    for i in range(vector_length)

]

nan_total = (

    df.select(
        sum(nan_exprs).alias("nan_total")
    )

    .collect()[0]["nan_total"]

)

# =========================================================
# +INF COUNTS
# =========================================================

pos_inf_exprs = [

    F.when(
        F.col("features_array")[i] == float("inf"),
        1
    ).otherwise(0)

    for i in range(vector_length)

]

pos_inf_total = (

    df.select(
        sum(pos_inf_exprs).alias("pos_inf_total")
    )

    .collect()[0]["pos_inf_total"]

)

# =========================================================
# -INF COUNTS
# =========================================================

neg_inf_exprs = [

    F.when(
        F.col("features_array")[i] == float("-inf"),
        1
    ).otherwise(0)

    for i in range(vector_length)

]

neg_inf_total = (

    df.select(
        sum(neg_inf_exprs).alias("neg_inf_total")
    )

    .collect()[0]["neg_inf_total"]

)

# =========================================================
# PRINT RESULTS
# =========================================================

print(f"Rows: {row_count}, "
      f"Vector length: {vector_length}, "
      f"Total values: {total_values}")

print(f"NaN total: {nan_total}")

print(f"+Inf total: {pos_inf_total}")

print(f"-Inf total: {neg_inf_total}")

# =========================================================
# PASS / FAIL
# =========================================================

if (
    nan_total > 0
    or pos_inf_total > 0
    or neg_inf_total > 0
):

    print(
        "GATE FAIL: "
        "Post-clean vectors still contain NaN/Inf."
    )

else:

    print(
        "GATE PASS: "
        "All feature vectors are finite."
    )

# =========================================================
# SAVE CSV REPORT
# =========================================================

report_df = spark.createDataFrame(

    [

        (
            "nan_total",
            int(nan_total)
        ),

        (
            "pos_inf_total",
            int(pos_inf_total)
        ),

        (
            "neg_inf_total",
            int(neg_inf_total)
        ),

        (
            "total_rows",
            int(row_count)
        ),

        (
            "vector_length",
            int(vector_length)
        ),

        (
            "total_feature_values",
            int(total_values)
        ),

    ],

    ["metric", "value"]

)

report_df.show(truncate=False)

report_df.coalesce(1).write.mode(
    "overwrite"
).option(
    "header",
    True
).csv(
    OUTPUT_CSV_PATH
)

print(
    "\nSaved post-clean validation CSV to:",
    OUTPUT_CSV_PATH
)

# =========================================================
# CLEANUP
# =========================================================

spark.stop()
