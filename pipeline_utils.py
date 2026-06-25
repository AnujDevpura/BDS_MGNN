import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_spark(
    app_name: str,
    master: Optional[str] = None,
    driver_memory: Optional[str] = None,
    shuffle_partitions: Optional[int] = None,
    default_parallelism: Optional[int] = None,
    enable_adaptive: bool = True,
    enable_dynamic_allocation: bool = False,
    parquet_compression: str = "zstd",
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)

    if master:
        builder = builder.master(master)
    if driver_memory:
        builder = builder.config("spark.driver.memory", driver_memory)
    if shuffle_partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if default_parallelism:
        builder = builder.config("spark.default.parallelism", str(default_parallelism))
    if enable_adaptive:
        builder = builder.config("spark.sql.adaptive.enabled", "true")
    if enable_dynamic_allocation:
        builder = (
            builder.config("spark.dynamicAllocation.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
        )

    builder = builder.config("spark.sql.parquet.compression.codec", parquet_compression)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def write_metrics_json(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def class_distribution(df: DataFrame, col_name: str = "attack_type") -> Dict[str, int]:
    rows = df.groupBy(col_name).count().collect()
    return {str(r[col_name]): int(r["count"]) for r in rows}


def validate_min_rows(df: DataFrame, min_rows: int, label: str) -> int:
    row_count = df.count()
    if row_count < min_rows:
        raise ValueError(f"{label} has {row_count} rows, below minimum required {min_rows}")
    return row_count


def validate_null_rate(
    df: DataFrame,
    col_name: str,
    max_null_rate: float,
    total_rows: Optional[int] = None,
) -> float:
    if total_rows is None:
        total_rows = df.count()
    null_rows = df.where(F.col(col_name).isNull()).count()
    null_rate = 0.0 if total_rows == 0 else null_rows / total_rows
    if null_rate > max_null_rate:
        raise ValueError(
            f"Column {col_name} null rate {null_rate:.6f} exceeds threshold {max_null_rate:.6f}"
        )
    return null_rate
