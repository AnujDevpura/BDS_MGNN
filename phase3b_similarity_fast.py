import argparse
import gc
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from pyspark.ml.functions import vector_to_array

from pipeline_utils import build_spark, write_metrics_json


def parse_args():
    p = argparse.ArgumentParser(description="Batched FAISS-IVF similarity graph")
    p.add_argument("--input", default="artifacts/phase2_sampled_500k")
    p.add_argument("--output", default="artifacts/phase3_similarity")
    p.add_argument("--metrics-output", default="artifacts/metrics/phase3b_metrics.json")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-similarity", type=float, default=0.80)
    p.add_argument("--nlist", type=int, default=256)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--search-batch-size", type=int, default=50000)
    p.add_argument("--faiss-threads", type=int, default=8)
    p.add_argument("--master", default=None)
    p.add_argument("--driver-memory", default="12g")
    p.add_argument("--shuffle-partitions", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark(
        "MGNN-Phase3B-FAISS-IVF", args.master, args.driver_memory,
        args.shuffle_partitions, 48, True, False, "zstd"
    )
    pdf = spark.read.parquet(args.input).select(
        "node_id", vector_to_array("features").alias("features")
    ).toPandas()
    spark.stop()
    node_ids = pdf["node_id"].to_numpy(dtype=np.int64)
    vectors = np.asarray(pdf["features"].tolist(), dtype=np.float32)
    del pdf
    gc.collect()
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    faiss.normalize_L2(vectors)
    faiss.omp_set_num_threads(args.faiss_threads)

    dim = vectors.shape[1]
    nlist = min(args.nlist, max(1, len(vectors) // 100))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(args.nprobe, nlist)

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output) if output.is_dir() else output.unlink()
    output.mkdir(parents=True)
    edge_count = 0

    for part, start in enumerate(range(0, len(vectors), args.search_batch_size)):
        end = min(start + args.search_batch_size, len(vectors))
        distances, indices = index.search(vectors[start:end], args.top_k + 1)
        src, dst, weight = [], [], []
        for local_i in range(end - start):
            source_idx = start + local_i
            added = 0
            for neighbor_idx, similarity in zip(indices[local_i], distances[local_i]):
                if neighbor_idx < 0 or neighbor_idx == source_idx or similarity < args.min_similarity:
                    continue
                src.append(int(node_ids[source_idx]))
                dst.append(int(node_ids[neighbor_idx]))
                weight.append(float(similarity))
                added += 1
                if added == args.top_k:
                    break
        edges = pd.DataFrame({"src": src, "dst": dst, "weight": weight, "edge_type": "similarity"})
        edges.drop_duplicates(["src", "dst"]).to_parquet(
            output / f"part-{part:05d}.parquet", index=False, compression="zstd"
        )
        edge_count += len(edges)
        print(f"Searched {end}/{len(vectors)} nodes | edges={edge_count}")

    write_metrics_json(args.metrics_output, {
        "phase": "phase3b_similarity", "backend": "faiss_ivf", "nodes": len(vectors),
        "edge_count": edge_count, "top_k": args.top_k, "nlist": nlist, "nprobe": index.nprobe,
    })
    print(f"Phase 3B complete | edges={edge_count} | output={output}")


if __name__ == "__main__":
    main()
