import json
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pipeline_utils import build_spark

def main():
    input_path = "artifacts/phase2_sampled_500k"
    partitions_to_test = [16, 32, 64, 128]
    results = []
    
    print("Starting Big Data Systems PySpark Distributed Shuffle Overhead Study")

    for p in partitions_to_test:
        print(f"Evaluating distributed computation with shuffle_partitions={p}")
        spark = build_spark(
            app_name=f"MGNN-Scaling-Study-{p}",
            master="local[*]",
            driver_memory="8g",
            shuffle_partitions=p,
            default_parallelism=p,
            enable_adaptive=False,  # Disable AQE so it respects our exact partition request
            enable_dynamic_allocation=False,
            parquet_compression="zstd",
        )
        
        # Suppress noisy logs
        spark.sparkContext.setLogLevel("ERROR")
        
        start_time = time.perf_counter()
        
        df = spark.read.parquet(input_path).select("node_id", "event_time")
        df = df.where(F.col("event_time").isNotNull())
        df = df.withColumn("time_bucket", (F.unix_timestamp("event_time") / 2).cast("long"))
        df = df.repartition("time_bucket")
        w = Window.partitionBy("time_bucket").orderBy("event_time")
        df = df.withColumn("row_num", F.row_number().over(w))

        left = df.alias("l")
        right = df.alias("r")
        cond = (
            (F.col("l.time_bucket") == F.col("r.time_bucket"))
            & (F.col("l.row_num") < F.col("r.row_num"))
            & (F.col("r.row_num") <= F.col("l.row_num") + 8)
        )
        
        edges = left.join(right, cond, "inner").withColumn(
            "time_diff",
            F.abs(F.unix_timestamp(F.col("r.event_time")) - F.unix_timestamp(F.col("l.event_time"))),
        )
        edges = edges.where(F.col("time_diff") <= 3)
        
        # Action to force distributed shuffle & compute graph
        edge_count = edges.count()
        
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        
        print(f"  Result: shuffle_partitions={p} -> {exec_time:.2f} seconds (Computed {edge_count} temporal edges)")
        
        results.append({
            "shuffle_partitions": p,
            "execution_time_seconds": float(exec_time),
            "edges_computed": int(edge_count)
        })
        
        spark.stop()
        time.sleep(2) # Give Spark memory time to flush
        
    out_path = Path("artifacts/research/pyspark_scaling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Scaling study successfully saved to {out_path}")

if __name__ == "__main__":
    main()
