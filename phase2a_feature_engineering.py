# phase2a_feature_engineering.py

# ============================================================
# MGNN PHASE 2A
# FEATURE ENGINEERING + MULTICLASS GRAPH FEATURES
# ============================================================

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StandardScaler,
    VectorAssembler,
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import (
    IntegerType,
    LongType,
    DoubleType,
    FloatType,
    ShortType,
    DecimalType,
)

# ============================================================
# CONFIG
# ============================================================

REPARTITION_COUNT = 32

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Phase 1 parquet path"
)

parser.add_argument(
    "--output",
    required=True,
    help="Phase 2 parquet output"
)

parser.add_argument(
    "--validation-output",
    required=True,
    help="Validation parquet output"
)

parser.add_argument(
    "--scale",
    action="store_true",
    help="Apply StandardScaler"
)

parser.add_argument("--master", default=None)
parser.add_argument("--driver-memory", default=None)
parser.add_argument("--partitions", type=int, default=REPARTITION_COUNT)

args = parser.parse_args()

# ============================================================
# SPARK
# ============================================================

builder = (
    SparkSession.builder
    .appName("MGNN-Phase2A")
    .config(
        "spark.sql.shuffle.partitions",
        args.partitions
    )
    .config(
        "spark.default.parallelism",
        args.partitions
    )
    .config(
        "spark.sql.execution.arrow.pyspark.enabled",
        "true"
    )
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.parquet.compression.codec", "zstd")
)
if args.master:
    builder = builder.master(args.master)
if args.driver_memory:
    builder = builder.config("spark.driver.memory", args.driver_memory)
spark = builder.getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\nSpark started")

# ============================================================
# LOAD PHASE 1
# ============================================================

print("\nLoading phase 1 parquet...")

df = spark.read.parquet(args.input)

print("Rows:", df.count())

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_cols = [

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

    "label_binary",

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

# ============================================================
# FEATURE COLUMN DETECTION
# ============================================================

excluded = {

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

    "label_binary",

    "partition_id",

    "replica_id",

}

numeric_cols = []

for field in df.schema.fields:

    if field.name in excluded:
        continue

    if isinstance(

        field.dataType,

        (
            IntegerType,
            LongType,
            DoubleType,
            FloatType,
            ShortType,
            DecimalType,
        )

    ):

        numeric_cols.append(field.name)

# deterministic ordering
numeric_cols = sorted(numeric_cols)

print("\nNumeric feature count:",
      len(numeric_cols))

print("\nExample numeric columns:")

for c in numeric_cols[:20]:
    print("-", c)

# ============================================================
# SIMPLE NUMERIC CLEANING
# ============================================================

print("\nCleaning numeric columns...")

INF_CAP = 1e6

for col_name in numeric_cols:

    c = F.col(col_name).cast("double")

    cleaned = (

        F.when(c.isNull(), 0.0)

        .when(F.isnan(c), 0.0)

        .when(c == float("inf"), INF_CAP)

        .when(c == float("-inf"), -INF_CAP)

        .otherwise(c)

    )

    df = df.withColumn(col_name, cleaned)

print("Numeric cleaning complete")

# ============================================================
# FEATURE ASSEMBLY
# ============================================================

print("\nAssembling features...")

assembler = VectorAssembler(

    inputCols=numeric_cols,

    outputCol="features_raw",

    handleInvalid="keep",

)

stages = [assembler]

feature_col = "features_raw"

# ============================================================
# OPTIONAL STANDARD SCALING
# ============================================================

if args.scale:

    print("\nApplying StandardScaler...")

    scaler = StandardScaler(

        inputCol="features_raw",

        outputCol="features",

        withMean=False,

        withStd=True,

    )

    stages.append(scaler)

    feature_col = "features"

# ============================================================
# PIPELINE
# ============================================================

print("\nBuilding Spark ML pipeline...")

pipeline = Pipeline(stages=stages)

print("\nFitting pipeline...")

model = pipeline.fit(df)

print("\nTransforming dataset...")

transformed = model.transform(df)

# ============================================================
# FINAL DATAFRAME
# ============================================================

print("\nCreating final dataframe...")

final_cols = [

    "node_id",

    "event_time",

    "attack_type",

    "label_multiclass",

    "label_binary",

]

if "replica_id" in transformed.columns:

    final_cols.append("replica_id")

final_cols.append(

    F.col(feature_col).alias("features")

)

final_df = transformed.select(*final_cols)

# ============================================================
# SAVE MAIN OUTPUT
# ============================================================

print("\nSaving Phase 2 parquet...")

final_df.write.mode(
    "overwrite"
).parquet(args.output)

# ============================================================
# SAVE VALIDATION SAMPLE
# ============================================================

print("\nSaving validation sample...")

validation_df = final_df.limit(1000)

validation_df.write.mode(
    "overwrite"
).parquet(args.validation_output)

# ============================================================
# STATS
# ============================================================

print("\n==============================")
print("PHASE 2A COMPLETE")
print("==============================")

print("\nOutput path:")
print(args.output)

print("\nValidation path:")
print(args.validation_output)

print("\nRows:")
print(final_df.count())

sample_row = final_df.select(
    "features"
).first()

if sample_row:

    print("\nFeature dimension:")
    print(sample_row["features"].size)

print("\nAttack distribution:")

final_df.groupBy(
    "attack_type"
).count().orderBy(
    F.desc("count")
).show(
    50,
    truncate=False
)

print("\nMulticlass distribution:")

final_df.groupBy(
    "label_multiclass"
).count().orderBy(
    "label_multiclass"
).show(
    50,
    truncate=False
)

print("\nSchema:")

final_df.printSchema()

print("\nSample rows:")

final_df.show(
    5,
    truncate=False
)

spark.stop()

print("\nPhase 2A complete.")
