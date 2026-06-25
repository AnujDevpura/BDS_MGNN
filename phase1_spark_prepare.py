# phase1_spark_prepare.py

# =========================================================
# MGNN PHASE 1
# RAW CICIDS PREPARATION + MULTICLASS LABEL ENGINEERING
# =========================================================

import argparse
import glob
import os
import re
from typing import List, Optional

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="MGNN Phase 1"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV glob"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path"
    )

    parser.add_argument(
        "--replication-factor",
        type=int,
        default=5
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=32
    )

    parser.add_argument("--master", default=None)
    parser.add_argument("--driver-memory", default=None)

    return parser.parse_args()

# =========================================================
# COLUMN NORMALIZATION
# =========================================================

def normalize_columns(df: DataFrame) -> DataFrame:

    cleaned = []
    seen = {}

    for c in df.columns:

        c2 = c.strip()

        c2 = c2.replace(" ", "_")
        c2 = c2.replace("/", "_")
        c2 = c2.replace("-", "_")
        c2 = c2.replace("(", "")
        c2 = c2.replace(")", "")

        c2 = re.sub(r"_+", "_", c2)

        # Handle duplicate column names
        if c2 in seen:

            seen[c2] += 1

            c2 = f"{c2}_{seen[c2]}"

        else:

            seen[c2] = 0

        cleaned.append(c2)

    return df.toDF(*cleaned)
# =========================================================
# REMOVE STRING WHITESPACE
# =========================================================

def trim_strings(df: DataFrame) -> DataFrame:

    exprs = []

    for name, dtype in df.dtypes:

        if dtype == "string":

            exprs.append(
                F.trim(F.col(name)).alias(name)
            )

        else:

            exprs.append(F.col(name))

    return df.select(*exprs)

# =========================================================
# FIND COLUMN CASE-INSENSITIVELY
# =========================================================

def resolve_column(
    df: DataFrame,
    requested: str
) -> Optional[str]:

    mapping = {
        c.lower(): c
        for c in df.columns
    }

    return mapping.get(requested.lower())

# =========================================================
# OPTIONAL SCALE SIMULATION
# =========================================================

def simulate_scale(
    df: DataFrame,
    replication_factor: int
) -> DataFrame:

    if replication_factor <= 1:

        return df

    copies = (
        df.sparkSession
        .range(0, replication_factor)
        .withColumnRenamed(
            "id",
            "replica_id"
        )
    )

    return df.crossJoin(copies)

# =========================================================
# SAFE TIMESTAMP HANDLING
# =========================================================

def ensure_timestamp(
    df: DataFrame,
    ts_col: Optional[str]
) -> DataFrame:

    if not ts_col:

        print(
            "\nWARNING: No timestamp column found."
        )

        return df.withColumn(
            "event_time",
            F.lit(None).cast("timestamp")
        )

    print("\nTimestamp column:",
          ts_col)

    raw = F.col(ts_col)
    return df.withColumn(
        "event_time",
        F.coalesce(
            F.to_timestamp(raw),
            F.to_timestamp(raw, "M/d/yyyy H:mm"),
            F.to_timestamp(raw, "M/d/yyyy H:mm:ss"),
            F.to_timestamp(raw, "dd/MM/yyyy HH:mm"),
            F.to_timestamp(raw, "dd/MM/yyyy HH:mm:ss"),
        )
    )

# =========================================================
# CANONICAL LABEL CLEANING
# =========================================================

def canonicalize_label(label):

    if label is None:
        return "unknown"

    label = str(label)

    label = label.strip().lower()

    # remove weird unicode chars
    label = label.encode(
        "ascii",
        "ignore"
    ).decode()

    # normalize separators
    label = label.replace("-", "_")

    label = label.replace(" ", "_")

    label = label.replace("/", "_")

    # remove duplicate underscores
    label = re.sub(r"_+", "_", label)

    # =====================================================
    # BENIGN
    # =====================================================

    if "benign" in label:
        return "benign"

    # =====================================================
    # DDOS
    # =====================================================

    if "ddos" in label:
        return "ddos"

    # =====================================================
    # DOS
    # =====================================================

    if "dos_hulk" in label:
        return "dos_hulk"

    if "goldeneye" in label:
        return "dos_goldeneye"

    if "slowloris" in label:
        return "dos_slowloris"

    if "slowhttptest" in label:
        return "dos_slowhttptest"

    # =====================================================
    # PORTSCAN
    # =====================================================

    if "portscan" in label:
        return "portscan"

    # =====================================================
    # BOT
    # =====================================================

    if "bot" in label:
        return "bot"

    # =====================================================
    # BRUTE FORCE
    # =====================================================

    if "ftp_patator" in label:
        return "ftp_patator"

    if "ssh_patator" in label:
        return "ssh_patator"

    # =====================================================
    # WEB ATTACKS
    # =====================================================

    if "sql" in label:
        return "web_sql_injection"

    if "xss" in label:
        return "web_xss"

    if "brute_force" in label:
        return "web_bruteforce"

    # =====================================================
    # INFILTRATION
    # =====================================================

    if "infiltration" in label:
        return "infiltration"

    # =====================================================
    # HEARTBLEED
    # =====================================================

    if "heartbleed" in label:
        return "heartbleed"

    return "unknown"

# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    builder = (
        SparkSession.builder
        .appName("MGNN-Phase1")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.default.parallelism", str(args.partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
    )
    if args.master:
        builder = builder.master(args.master)
    if args.driver_memory:
        builder = builder.config("spark.driver.memory", args.driver_memory)
    spark = builder.getOrCreate()

    print("\nSpark started")

    # =====================================================
    # INPUT FILES
    # =====================================================

    input_paths: List[str]

    if any(
        token in args.input
        for token in ["*", "?", "["]
    ):

        input_paths = sorted(
            glob.glob(args.input)
        )

    elif os.path.isdir(args.input):

        input_paths = sorted(
            glob.glob(
                os.path.join(
                    args.input,
                    "*.csv"
                )
            )
        )

    else:

        input_paths = [args.input]

    if not input_paths:

        raise FileNotFoundError(
            f"No CSV files matched: {args.input}"
        )

    print("\nCSV files found:",
          len(input_paths))

    # =====================================================
    # LOAD RAW CSV
    # =====================================================

    print("\nLoading CICIDS CSVs...")

    raw_df = (
        spark.read
        .option("header", True)
        # CICIDS files contain duplicate and whitespace-inconsistent headers.
        # Apply the inferred schema by column position instead of validating names per file.
        .option("enforceSchema", True)
        .option("mode", "PERMISSIVE")
        .option("inferSchema", True)
        .csv(input_paths)
    )

    # =====================================================
    # CLEAN COLUMNS
    # =====================================================

    print("\nCleaning columns...")

    df = normalize_columns(raw_df)

    df = trim_strings(df)

    # =====================================================
    # LABEL COLUMN
    # =====================================================

    label_col = resolve_column(
        df,
        "Label"
    )

    if not label_col:

        raise ValueError(
            "Could not find Label column"
        )

    print("\nLabel column:",
          label_col)

    # =====================================================
    # CLEAN LABELS
    # =====================================================

    print("\nCanonicalizing labels...")

    canonical_udf = F.udf(
        canonicalize_label,
        StringType()
    )

    df = df.withColumn(
        "attack_type",
        canonical_udf(
            F.col(label_col)
        )
    )

    # =====================================================
    # MULTICLASS LABEL IDS
    # =====================================================

    print("\nCreating multiclass IDs...")

    label_order = [
        "benign",
        "ddos",
        "dos_hulk",
        "dos_goldeneye",
        "dos_slowloris",
        "dos_slowhttptest",
        "portscan",
        "bot",
        "ftp_patator",
        "ssh_patator",
        "web_sql_injection",
        "web_xss",
        "web_bruteforce",
        "infiltration",
        "heartbleed",
        "unknown"
    ]

    mapping_expr = F.create_map([
        F.lit(x)
        for kv in enumerate(label_order)
        for x in kv[::-1]
    ])

    df = df.withColumn(
        "label_multiclass",
        mapping_expr[
            F.col("attack_type")
        ]
    )

    # =====================================================
    # OPTIONAL BINARY LABEL
    # =====================================================

    df = df.withColumn(
        "label_binary",
        F.when(
            F.col("attack_type") == "benign",
            0
        ).otherwise(1)
    )

    # =====================================================
    # TIMESTAMP DETECTION
    # =====================================================

    possible_timestamp_cols = []

    for c in df.columns:

        lower = c.lower()

        if (
            "time" in lower
            or "timestamp" in lower
            or "date" in lower
        ):

            possible_timestamp_cols.append(c)

    print("\nPossible timestamp columns:")

    for c in possible_timestamp_cols:
        print("-", c)

    timestamp_col = (
        possible_timestamp_cols[0]
        if possible_timestamp_cols
        else None
    )

    # =====================================================
    # SAFE TIMESTAMP PARSING
    # =====================================================

    df = ensure_timestamp(
        df,
        timestamp_col
    )

    # =====================================================
    # FALLBACK TIMESTAMPS
    # =====================================================

    df = df.withColumn(
        "event_time",
        F.coalesce(
            F.col("event_time"),
            F.to_timestamp(
                F.from_unixtime(
                    F.lit(1700000000)
                    +
                    (
                        F.monotonically_increasing_id()
                        % F.lit(86400)
                    )
                )
            )
        )
    )

    # =====================================================
    # OPTIONAL SCALE SIMULATION
    # =====================================================

    print(
        "\nApplying replication factor:",
        args.replication_factor
    )

    df = simulate_scale(
        df,
        args.replication_factor
    )

    # =====================================================
    # NODE IDS
    # =====================================================

    print("\nCreating node IDs...")

    df = df.withColumn(
        "node_id",
        F.monotonically_increasing_id()
    )

    # =====================================================
    # REPARTITION
    # =====================================================

    print("\nRepartitioning...")

    df = df.repartition(args.partitions)

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    final_df = df.withColumn(
        "partition_id",
        F.spark_partition_id()
    )

    # =====================================================
    # SAVE
    # =====================================================

    print("\nSaving parquet...")

    final_df.write.mode(
        "overwrite"
    ).parquet(args.output)

    # =====================================================
    # STATS
    # =====================================================

    print("\n==============================")
    print("PHASE 1 COMPLETE")
    print("==============================")

    total_rows = final_df.count()

    print("\nRows:", total_rows)

    print(
        "Partitions:",
        final_df.rdd.getNumPartitions()
    )

    print("\n==============================")
    print("ATTACK DISTRIBUTION")
    print("==============================")

    final_df.groupBy(
        "attack_type"
    ).count().orderBy(
        F.desc("count")
    ).show(
        50,
        truncate=False
    )

    print("\n==============================")
    print("MULTICLASS DISTRIBUTION")
    print("==============================")

    final_df.groupBy(
        "label_multiclass"
    ).count().orderBy(
        "label_multiclass"
    ).show(
        50,
        truncate=False
    )

    print("\n==============================")
    print("BINARY DISTRIBUTION")
    print("==============================")

    final_df.groupBy(
        "label_binary"
    ).count().show()

    print("\nOutput:")
    print(args.output)

    spark.stop()

# =========================================================

if __name__ == "__main__":
    main()
